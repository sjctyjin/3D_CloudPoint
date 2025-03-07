import numpy as np
import cv2
import struct

# 讀取 RGB 和深度圖像
rgb_path = "rgb_image.png"
depth_path = "depth_image.png"

rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # 轉換為RGB格式
depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)  # 讀取灰階深度圖

# 獲取圖像尺寸
height, width = depth_image.shape

# 假設相機內參 (簡單設置)
fx = fy = max(width, height) * 1.2  # 假設焦距
cx, cy = width / 2, height / 2  # 光心位置

# 生成 3D 點雲
points = []
colors = []

for v in range(height):
    for u in range(width):
        depth = depth_image[v, u] / 255.0 * 5.0  # 假設最大深度為5米
        if depth > 0:
            x = (u - cx) * depth / fx
            y = (v - cy) * depth / fy
            z = depth
            points.append((x, y, z))
            colors.append(rgb_image[v, u])  # RGB 顏色

# 轉換為 numpy 陣列
points = np.array(points, dtype=np.float32)
colors = np.array(colors, dtype=np.uint8)

# 保存為 PLY 格式
ply_path = "output_point_cloud.ply"

with open(ply_path, "wb") as ply_file:
    # 寫入 PLY 頭部
    ply_file.write(b"ply\n")
    ply_file.write(b"format binary_little_endian 1.0\n")
    ply_file.write(f"element vertex {len(points)}\n".encode())
    ply_file.write(b"property float x\nproperty float y\nproperty float z\n")
    ply_file.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
    ply_file.write(b"end_header\n")

    # 寫入數據
    for i in range(len(points)):
        ply_file.write(struct.pack("fffBBB", *points[i], *colors[i]))

# 返回 PLY 文件的路徑，供下載
# ply_path
