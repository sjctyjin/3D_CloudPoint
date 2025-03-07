import numpy as np
import open3d as o3d
import copy
import os


def create_gripper_geometry(width=0.08, depth=0.06):
    """
    创建一个简化的平行夹爪几何模型，开度与最窄距离匹配

    Args:
        width: 夹爪宽度（夹爪打开的距离）
        depth: 夹爪深度

    Returns:
        gripper: 夹爪几何模型的列表
    """
    # 创建夹爪的基座
    base = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.02, depth=0.02)
    base.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
    base.translate([-0.01, -0.01, 0])

    # 创建夹爪的左指
    left_finger = o3d.geometry.TriangleMesh.create_box(width=width*0.1, height=width / 2, depth=width)
    left_finger.paint_uniform_color([1, 0.5, 0])  # 橙色
    left_finger.translate([width-0.05, 0, 0])

    # 创建夹爪的右指
    right_finger = o3d.geometry.TriangleMesh.create_box(width=width*0.1, height=width / 2, depth=width)
    right_finger.paint_uniform_color([1, 1, 0])  # 黃色
    right_finger.translate([-width+0.05, 0, 0])

    # 创建夹爪的坐标系
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    # 返回夹爪几何模型的列表
    return [base, left_finger, right_finger, frame]


def generate_grasp_pose(point_cloud):
    """
    从点云生成抓取姿态，考虑物体在XY平面上的最窄位置，夹爪垂直于最窄线

    Args:
        point_cloud: Open3D点云对象

    Returns:
        grasp_position: 抓取位置 (3D向量)
        rotation_matrix: 抓取方向 (3x3旋转矩阵)
        narrow_width: 最窄方向的宽度
        grasp_points: 最窄方向上的两个极点
    """
    # 获取点云数据
    points = np.asarray(point_cloud.points)
    if len(points) < 10:
        print('点云中点数太少，无法生成可靠的抓取姿态')
        return None, None, None, None

    # 计算点云的中心
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    # 对点云进行主成分分析
    cov_matrix = np.cov(points_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 对特征值/特征向量进行排序（从小到大）
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 提取主成分轴
    minor_axis = eigenvectors[:, 0]  # 最小主成分
    middle_axis = eigenvectors[:, 1]  # 中间主成分
    major_axis = eigenvectors[:, 2]  # 最大主成分

    # 估计物体尺寸
    obj_size = np.max(np.abs(points_centered)) * 2

    # 物体形状分析
    shape_elongation = eigenvalues[2] / eigenvalues[1]
    shape_flatness = eigenvalues[1] / eigenvalues[0]
    print(f"形状分析 - 延展度: {shape_elongation:.2f}, 平坦度: {shape_flatness:.2f}")
    print(f"物体尺寸估计: {obj_size:.3f} m")

    # 寻找XY平面上的最窄位置
    best_width = float('inf')
    best_angle = 0

    # 在XY平面上搜索不同角度
    angle_steps = 36  # 每5度一步
    for i in range(angle_steps):
        angle = i * (np.pi / (angle_steps / 2))  # 0到180度
        direction = np.array([np.cos(angle), np.sin(angle), 0])

        # 将点投影到这个方向
        proj = np.dot(points_centered, direction)
        width = np.max(proj) - np.min(proj)

        if width < best_width:
            best_width = width
            best_angle = angle

    # 计算最窄方向和垂直于它的方向
    narrow_direction = np.array([np.cos(best_angle), np.sin(best_angle), 0])
    narrow_direction = narrow_direction / np.linalg.norm(narrow_direction)
    perpendicular_direction = np.array([-np.sin(best_angle), np.cos(best_angle), 0])
    perpendicular_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)

    print(f"找到XY平面上的最窄方向，角度: {best_angle * 180 / np.pi:.1f}度, 宽度: {best_width:.3f}m")

    # 决定抓取姿态 - 夹爪垂直于最窄线

    # 最窄线是在XY平面上的，我们需要一个垂直于XY平面的向量作为z轴（接近方向）
    z_axis = np.array([0, 0, 1])  # 垂直于XY平面的向量

    # x轴（夹爪张开方向）应该沿着最窄方向
    x_axis = narrow_direction

    # y轴通过叉积计算，确保坐标系正交
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # 构建旋转矩阵
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    # 寻找合适的抓取点
    # 在最窄方向上找到两个极点
    proj = np.dot(points, narrow_direction)
    min_idx = np.argmin(proj)
    max_idx = np.argmax(proj)

    # 获取最窄方向上的两个极点
    min_point = points[min_idx]
    max_point = points[max_idx]

    # 计算极点之间的精确距离
    narrow_width = np.linalg.norm(max_point - min_point)

    # 抓取点应该在两个极点的中间上方
    midpoint = (min_point + max_point) / 2

    # 计算抓取位置（在中点上方）
    offset_distance = obj_size * 0.2  # 上方偏移物体尺寸的20%
    grasp_position = midpoint + z_axis * offset_distance

    # 确保我们使用的最窄宽度是从点云实际测量的
    print(f"最窄部位的实际宽度: {narrow_width:.3f}m")

    return grasp_position, rotation_matrix, narrow_width, [min_point, max_point]


def visualize_grasp(point_cloud, grasp_position, rotation_matrix, narrow_width, grasp_points=None):
    """
    可视化点云和抓取姿态

    Args:
        point_cloud: Open3D点云对象
        grasp_position: 抓取位置
        rotation_matrix: 抓取方向（旋转矩阵）
        narrow_width: 最窄方向的宽度，用于设置夹爪开度
        grasp_points: 最窄方向上的两个极点
    """
    # 创建夹爪视觉模型 - 使用最窄宽度作为夹爪开度
    # 增加一点余量，确保夹爪能完全包住物体
    gripper_width = narrow_width * 1.1  # 增加10%余量
    gripper_depth = narrow_width * 1.5  # 夹爪深度设为宽度的1.5倍
    print(gripper_width)
    gripper_parts = create_gripper_geometry(width=gripper_width, depth=gripper_depth)

    # 设置夹爪位置和方向
    geometries = []
    for part in gripper_parts:
        transformed_part = copy.deepcopy(part)
        transformed_part.rotate(rotation_matrix)
        transformed_part.translate(grasp_position)
        geometries.append(transformed_part)

    # 创建一个包含原点的坐标系
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=narrow_width)
    geometries.append(origin)

    # 设置点云颜色
    point_cloud_colored = copy.deepcopy(point_cloud)
    point_cloud_colored.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色点云
    geometries.append(point_cloud_colored)

    # 显示抓取时夹爪到物体中心的线
    centroid = np.mean(np.asarray(point_cloud.points), axis=0)
    line_points = np.array([grasp_position, centroid])
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(line_points)
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色线
    geometries.append(line)

    # 显示最窄位置的两个极点和连接线
    if grasp_points is not None:
        # 添加极点（红色球体）
        for point in grasp_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=narrow_width * 0.05)
            sphere.paint_uniform_color([1, 0, 0])  # 红色
            sphere.translate(point)
            geometries.append(sphere)

        # 连接两个极点的线（表示最窄位置）- 绿色线
        line_points = np.array(grasp_points)
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(line_points)
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # 绿色线
        geometries.append(line)

    # 可视化
    o3d.visualization.draw_geometries(geometries,
                                      window_name="抓取姿态可视化",
                                      width=800, height=600)


def main():
    print("正在读取点云文件...")

    # 请替换为您的PLY文件路径
    file_path = "bun000.ply"

    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 读取点云
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"点云加载完成，共有 {len(point_cloud.points)} 个点")

    # 展示原始点云
    print("显示原始点云...")
    o3d.visualization.draw_geometries([point_cloud],
                                      window_name="原始点云",
                                      width=800, height=600)

    # 生成抓取姿态
    print("计算最佳抓取姿态...")
    result = generate_grasp_pose(point_cloud)

    if result is not None:
        grasp_position, rotation_matrix, narrow_width, grasp_points = result
        # 展示抓取姿态
        print("抓取姿态生成完成，正在可视化...")
        print(f"抓取位置: {grasp_position}")
        print(f"抓取方向: \n{rotation_matrix}")
        print(f"夹爪开度: {narrow_width:.3f}m")
        visualize_grasp(point_cloud, grasp_position, rotation_matrix, narrow_width, grasp_points)
    else:
        print("无法生成抓取姿态")


if __name__ == "__main__":
    main()