import numpy as np
import open3d as o3d
import cv2
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

    return [camera_frame, pyramid]


def create_pointcloud_from_rgbd(color_image, depth_image, intrinsic, depth_scale=1000.0):
    """從RGB和深度圖像創建點雲"""
    # 創建Open3D格式的彩色和深度圖像
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_image)

    # 打印深度圖信息
    print(f"深度圖尺寸: {depth_image.shape}")
    print(f"深度圖數據類型: {depth_image.dtype}")
    print(f"深度圖範圍: {np.min(depth_image)} - {np.max(depth_image)}")

    # 創建RGBD圖像
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,  # 深度縮放因子 (1000.0為毫米轉米)
        depth_trunc=5.0,  # 截斷5米外的深度
        convert_rgb_to_intensity=False
    )

    # 創建內參矩陣
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=color_image.shape[1],
        height=color_image.shape[0],
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2]
    )

    # 創建點雲
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic_o3d
    )

    # 檢查生成的點雲
    print(f"生成了 {len(pcd.points)} 個點")

    # 移除離群點，使點雲更乾淨
    if len(pcd.points) > 100:  # 確保有足夠的點進行統計濾波
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"過濾後還有 {len(pcd.points)} 個點")

    return pcd


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


def visualize_pointcloud_with_camera():
    """從RGB和深度圖像創建點雲，並與相機原點一起可視化"""
    # 加載圖像
    color_image_path = "rgb_image.png"  # 使用.png格式
    depth_image_path = "depth_image.png"

    # 檢查文件是否存在
    if not os.path.exists(color_image_path):
        print(f"錯誤：找不到RGB圖像 {color_image_path}")
        return
    if not os.path.exists(depth_image_path):
        print(f"錯誤：找不到深度圖像 {depth_image_path}")
        return

    # 讀取圖像
    color_image = cv2.imread(color_image_path)
    # 深度圖通常是16位的，確保使用正確的讀取方式
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 檢查圖像是否讀取成功
    if color_image is None:
        print(f"無法讀取RGB圖像 {color_image_path}")
        return
    if depth_image is None:
        print(f"無法讀取深度圖像 {depth_image_path}")
        return

    # 打印圖像信息
    print(f"RGB圖像尺寸: {color_image.shape}")
    print(f"深度圖像尺寸: {depth_image.shape}")

    # 假設您的相機內參
    # 注意：請使用您實際相機的內參值替換這些值
    intrinsic = np.array([
        [433.56628418, 0, 327.215271],
        [0, 432.98043823, 244.18916321],
        [0, 0, 1]
    ])

    # 創建點雲
    # 注意：depth_scale參數表示深度值到米的轉換因子
    # 對於以毫米為單位的深度圖，應該是1000.0
    # 如果您的深度圖是以不同單位存儲的，請調整此值
    pcd = create_pointcloud_from_rgbd(color_image, depth_image, intrinsic, depth_scale=1000.0)

    # 檢查點雲是否成功創建
    if pcd is None or len(pcd.points) == 0:
        print("生成的點雲為空或無效")
        return

    # 獲取點雲的邊界框以設置適當的相機視覺化尺寸
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    scale = max(bbox_extent) * 0.1  # 相機大小為點雲尺寸的10%

    # 創建相機可視化對象，位於原點(0,0,0)
    camera_geometries = create_camera_visualization(scale=scale)

    # 創建一個包含點雲和相機的場景
    # 這裡也可以使用draw_geometries函數，但使用Visualizer可以獲取更多控制
    geometries = [pcd] + camera_geometries

    # 創建可視化器
    vis = o3d.visualization.Visualizer()
    vis.create_window("點雲與相機位置", width=1024, height=768)

    # 添加點雲和相機幾何體
    for geom in geometries:
        vis.add_geometry(geom)

    # 設置更好的視角
    view_control = vis.get_view_control()
    # 重置視點
    vis.reset_view_point(True)

    # 使相機看向點雲的中心
    center = pcd.get_center()

    # 調整視點使得相機原點和點雲都可見
    # 這需要相機視點在相機原點和點雲之間
    offset = center * 0.5  # 在原點和點雲中心之間
    view_control.set_lookat(offset)

    # 運行可視化
    print("顯示點雲和相機原點(0,0,0)，關閉窗口以繼續...")
    vis.run()

    # 獲取用戶調整後的視角參數
    view_params = get_view_parameters(vis)
    vis.destroy_window()

    # 打印視角參數
    print("視角參數:")
    print(f"位置: {view_params['position']}")
    print(f"旋轉矩陣:\n{view_params['rotation']}")

    return view_params


if __name__ == "__main__":
    # 執行可視化
    view_params = visualize_pointcloud_with_camera()