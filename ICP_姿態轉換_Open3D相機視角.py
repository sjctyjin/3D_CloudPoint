import numpy as np
import open3d as o3d
import copy
import os

def create_camera_visualization(scale=0.1):
    """創建表示相機的幾何體"""
    # 創建相機坐標系
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)

    # 創建一個金字塔形狀表示相機視場
    pyramid_points = np.array([
        [0, 0, 0],  # 相機中心
        [scale, scale, scale * 2],  # 右上
        [-scale, scale, scale * 2],  # 左上
        [-scale, -scale, scale * 2],  # 左下
        [scale, -scale, scale * 2]  # 右下
    ])

    pyramid_lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # 從中心到各角
        [1, 2], [2, 3], [3, 4], [4, 1]  # 連接角點
    ])

    # 創建線條集合
    pyramid = o3d.geometry.LineSet()
    pyramid.points = o3d.utility.Vector3dVector(pyramid_points)
    pyramid.lines = o3d.utility.Vector2iVector(pyramid_lines)
    pyramid.paint_uniform_color([1, 0, 0])  # 紅色相機

    return camera_frame, pyramid

def get_view_parameters(vis):
    """
    獲取Open3D視窗中的視角參數

    Args:
        vis: Open3D可視化視窗

    Returns:
        view_params: 視角參數字典
    """
    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()

    # 獲取相機外部參數（位置和方向）
    extrinsic = cam_params.extrinsic

    # 從外部參數矩陣中提取旋轉和平移
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]

    return {
        "position": translation,
        "rotation": rotation,
        "intrinsic": cam_params.intrinsic.intrinsic_matrix,
        "extrinsic": extrinsic
    }

def create_gripper_geometry(width=0.08, depth=0.06):
    """
    創建一個簡化的平行夾爪幾何模型，開度與最窄距離匹配

    Args:
        width: 夾爪寬度（夾爪打開的距離）
        depth: 夾爪深度

    Returns:
        gripper: 夾爪幾何模型的列表
    """
    # 創建夾爪的基座
    base = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.02, depth=0.02)
    base.paint_uniform_color([0.8, 0.8, 0.8])  # 淺灰色
    base.translate([-0.01, -0.01, 0])

    # 创建夹爪的左指
    left_finger = o3d.geometry.TriangleMesh.create_box(width=width * 0.1, height=width / 2, depth=width)
    left_finger.paint_uniform_color([1, 0.5, 0])  # 橙色
    left_finger.translate([width - 0.05, 0, 0])

    # 创建夹爪的右指
    right_finger = o3d.geometry.TriangleMesh.create_box(width=width * 0.1, height=0.02, depth=width)
    right_finger.paint_uniform_color([1, 1, 0])  # 黃色
    right_finger.translate([-width + 0.05, 0, 0])

    # 創建夾爪的坐標系
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    # 返回夾爪幾何模型的列表
    return [base, left_finger, right_finger, frame]


def get_view_parameters(vis):
    """
    獲取Open3D視窗中的視角參數

    Args:
        vis: Open3D可視化視窗

    Returns:
        view_params: 視角參數字典
    """
    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()

    # 獲取相機外部參數（位置和方向）
    extrinsic = cam_params.extrinsic

    # 從外部參數矩陣中提取旋轉和平移
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]

    return {
        "position": translation,
        "rotation": rotation,
        "intrinsic": cam_params.intrinsic.intrinsic_matrix
    }


def generate_grasp_pose(point_cloud, view_params=None):
    """
    從點雲生成抓取姿態，考慮物體在XY平面上的最窄位置，夾爪垂直於最窄線
    以Open3D視角為參考點

    Args:
        point_cloud: Open3D點雲對象
        view_params: Open3D視角參數，包含位置和旋轉

    Returns:
        grasp_position: 抓取位置 (3D向量)，相對於全局坐標系
        rotation_matrix: 抓取方向 (3x3旋轉矩陣)
        narrow_width: 最窄方向的寬度
        grasp_points: 最窄方向上的兩個極點
        view_params: 視角參數，用於可視化
    """
    # 獲取點雲數據
    points = np.asarray(point_cloud.points)
    if len(points) < 10:
        print('點雲中點數太少，無法生成可靠的抓取姿態')
        return None, None, None, None, None

    # 如果提供了視角參數，則使用它們來轉換坐標系
    camera_position = np.array([0, 0, 0])
    camera_rotation = np.eye(3)  # 單位矩陣（無旋轉）

    if view_params is not None:
        camera_position = view_params["position"]
        camera_rotation = view_params["rotation"]
        print(f"使用Open3D視角作為參考：")
        print(f"相機位置: {camera_position}")
        print(f"相機方向: \n{camera_rotation}")

    # 計算點雲的中心
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    # 對點雲進行主成分分析
    cov_matrix = np.cov(points_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 對特徵值/特徵向量進行排序（從小到大）
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 提取主成分軸
    minor_axis = eigenvectors[:, 0]  # 最小主成分
    middle_axis = eigenvectors[:, 1]  # 中間主成分
    major_axis = eigenvectors[:, 2]  # 最大主成分

    # 估計物體尺寸
    obj_size = np.max(np.abs(points_centered)) * 2

    # 物體形狀分析
    shape_elongation = eigenvalues[2] / eigenvalues[1]
    shape_flatness = eigenvalues[1] / eigenvalues[0]
    print(f"形狀分析 - 延展度: {shape_elongation:.2f}, 平坦度: {shape_flatness:.2f}")
    print(f"物體尺寸估計: {obj_size:.3f} m")

    # 以視角方向為參考，定義XY平面
    # 假設視角的Z軸是指向相機的方向
    view_z_axis = camera_rotation[:, 2]  # 相機的Z軸

    # 找到與視角Z軸垂直的平面上的最窄位置
    best_width = float('inf')
    best_direction = None

    # 在與視角Z軸垂直的平面上搜索不同角度
    angle_steps = 36  # 每5度一步
    for i in range(angle_steps):
        angle = i * (np.pi / (angle_steps / 2))  # 0到180度

        # 在與視角Z軸垂直的平面上生成方向向量
        # 首先在XY平面上生成向量，然後旋轉到與視角Z軸垂直的平面
        xy_direction = np.array([np.cos(angle), np.sin(angle), 0])

        # 確保方向向量與視角Z軸垂直
        direction = xy_direction - np.dot(xy_direction, view_z_axis) * view_z_axis
        direction = direction / np.linalg.norm(direction)

        # 將點投影到這個方向
        proj = np.dot(points_centered, direction)
        width = np.max(proj) - np.min(proj)

        if width < best_width:
            best_width = width
            best_direction = direction

    # 計算與最窄方向垂直且垂直於視角Z軸的方向
    narrow_direction = best_direction
    perp_direction = np.cross(view_z_axis, narrow_direction)
    perp_direction = perp_direction / np.linalg.norm(perp_direction)

    print(f"找到最窄方向，寬度: {best_width:.3f}m")

    # 決定抓取姿態 - 夾爪垂直於最窄線且從視角方向接近

    # z軸（接近方向）應該與視角Z軸相反，即從相機向物體的方向
    z_axis = -view_z_axis  # 從相機指向物體

    # x軸（夾爪張開方向）應該沿著最窄方向
    x_axis = narrow_direction

    # y軸通過叉積計算，確保坐標系正交
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 重新計算x軸以確保正交性
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # 構建旋轉矩陣
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # 尋找合適的抓取點
    # 在最窄方向上找到兩個極點
    proj = np.dot(points, narrow_direction)
    min_idx = np.argmin(proj)
    max_idx = np.argmax(proj)

    # 獲取最窄方向上的兩個極點
    min_point = points[min_idx]
    max_point = points[max_idx]

    # 計算極點之間的精確距離
    narrow_width = np.linalg.norm(max_point - min_point)

    # 抓取點應該在兩個極點的中間，並從視角方向接近
    midpoint = (min_point + max_point) / 2

    # 計算從相機到中點的方向向量
    cam_to_mid = midpoint - camera_position
    cam_to_mid_norm = np.linalg.norm(cam_to_mid)

    # 計算抓取位置
    # 從相機方向稍微偏移物體表面
    offset_ratio = 0.2  # 相對於相機到物體的距離
    grasp_position = midpoint - z_axis * (cam_to_mid_norm * offset_ratio)

    # 確保我們使用的最窄寬度是從點雲實際測量的
    print(f"最窄部位的實際寬度: {narrow_width:.3f}m")

    return grasp_position, rotation_matrix, narrow_width, [min_point, max_point], {
        "position": camera_position,
        "rotation": camera_rotation
    }


def visualize_with_grasp(point_cloud,scene_select=True,view_params = None):
    """
    先顯示點雲，獲取用戶選擇的視角，然後計算並顯示抓取姿態

    Args:
        point_cloud: Open3D點雲對象
    """
    if scene_select:
        # 第一步：顯示點雲並讓用戶調整視角
        print("請調整視角到您想要的位置，然後關閉視窗...")

        # 創建可視化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="調整視角", width=800, height=600)
        vis.add_geometry(point_cloud)

        # 等待用戶調整視角
        vis.run()

        # 獲取視角參數
        view_params = get_view_parameters(vis)
        vis.destroy_window()
    print("視角調整完成 : ",view_params)
    # 第二步：計算抓取姿態
    print("基於選擇的視角計算抓取姿態...")
    if view_params is None:
        print("無法生成抓取姿態")
        return
    result = generate_grasp_pose(point_cloud, view_params)

    if result is None:
        print("無法生成抓取姿態")
        return

    grasp_position, rotation_matrix, narrow_width, grasp_points, view_params = result

    # 第三步：顯示點雲和抓取姿態
    print("顯示抓取姿態...")

    # 創建夾爪視覺模型
    gripper_width = narrow_width * 1.1  # 增加10%余量
    gripper_depth = narrow_width * 1.5  # 夾爪深度設為寬度的1.5倍
    gripper_parts = create_gripper_geometry(width=gripper_width, depth=gripper_depth)

    # 設置夾爪位置和方向
    geometries = []
    for part in gripper_parts:
        transformed_part = copy.deepcopy(part)
        transformed_part.rotate(rotation_matrix)
        transformed_part.translate(grasp_position)
        geometries.append(transformed_part)
        # 創建相機可視化對象
    camera_frame_work, pyramid = create_camera_visualization(scale=0.1)  # 調整尺寸以適應您的點雲
    # 創建一個表示相機位置的坐標系
    # camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=narrow_width * 0.5)
    # camera_frame.rotate(view_params["rotation"])
    # camera_frame.translate(view_params["position"])
    # geometries.append(camera_frame)
    geometries.append(pyramid)
    geometries.append(camera_frame_work)

    # 連接相機和抓取點的線
    line_points = np.array([view_params["position"], grasp_position])
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(line_points)
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # 藍色線
    geometries.append(line)

    # 最窄位置的兩個極點和連接線
    if grasp_points is not None:
        # 添加極點（紅色球體）
        for point in grasp_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=narrow_width * 0.05)
            sphere.paint_uniform_color([1, 0, 0])  # 紅色
            sphere.translate(point)
            geometries.append(sphere)

        # 連接兩個極點的線（表示最窄位置）- 綠色線
        line_points = np.array(grasp_points)
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(line_points)
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # 綠色線
        geometries.append(line)

    # 設置點雲顏色
    point_cloud_colored = copy.deepcopy(point_cloud)
    point_cloud_colored.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色點雲
    geometries.append(point_cloud_colored)

    # 創建可視化器
    # vis = o3d.visualization.Visualizer()
    # vis.create_window("點雲與相機位置", width=1024, height=768)
    # 顯示結果
    o3d.visualization.draw_geometries(geometries,
                                      window_name="抓取姿態可視化",
                                      width=800, height=600)

    # 打印抓取信息
    print("抓取姿態生成完成：")
    print(f"抓取位置: {grasp_position}")
    print(f"抓取方向: \n{rotation_matrix}")
    print(f"夾爪開度: {narrow_width:.3f}m")

    return grasp_position, rotation_matrix, narrow_width


def main():
    print("正在讀取點雲文件...")

    # 請替換為您的PLY文件路徑
    file_path = "output_point_cloud.ply"

    if not os.path.exists(file_path):
        print(f"錯誤: 找不到文件 {file_path}")
        return

    # 讀取點雲
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"點雲加載完成，共有 {len(point_cloud.points)} 個點")

    # 使用互動式視角來計算抓取姿態
    visualize_with_grasp(point_cloud)


if __name__ == "__main__":
    main()