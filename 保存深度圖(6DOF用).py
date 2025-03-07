import pyrealsense2 as rs
import numpy as np
import cv2

# 配置 RealSense 流程管線
pipeline = rs.pipeline()
config = rs.config()

# 開啟彩色和深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 開始拍攝
profile = pipeline.start(config)

# 獲取相機的內參
# 獲取顏色流的配置文件
color_profile = profile.get_stream(rs.stream.color)
# 將配置文件轉換為視頻流配置文件以訪問內參
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# 創建相機內參矩陣
camera_intrinsic = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
])


print("相機內參矩陣:")
print(camera_intrinsic)

# 顯示其他內參參數
print("\n其他相機參數:")
print(f"型號: {color_intrinsics.model}")
print(f"畸變係數: {color_intrinsics.coeffs}")
print(f"寬度: {color_intrinsics.width}")
print(f"高度: {color_intrinsics.height}")


try:
    print("拍攝中，請稍等...")
    # 等待捕捉一個幀
    frames = pipeline.wait_for_frames()

    # 獲取彩色和深度幀
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        raise Exception("未成功捕捉到幀，請檢查相機連接。")

    # 將幀轉換為numpy數組
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 將深度圖像縮放為可視範圍
    # depth_image_scaled = cv2.convertScaleAbs(depth_image, alpha=0.03)
    # 將深度圖數據歸一化到 0-255 範圍
    normalized_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_depth_image = np.uint8(normalized_depth_image)


    # 保存為 PNG 圖片
    cv2.imwrite("rgb_image.png", color_image)  # 保存RGB圖像
    cv2.imwrite("depth_image.png", normalized_depth_image)  # 保存深度圖像
    depth_path = "depth_image.png"
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 假設最大深度範圍為 0-10000 mm，映射回16位
    depth_image_16bit = (depth_image.astype(np.float32) / 255.0 * 10000).astype(np.uint16)
    cv2.imwrite("depth_image_16bit.png", depth_image_16bit)
    print("圖片已成功保存為 'rgb_image.png' 和 'depth_image.png'")
finally:
    # 停止相機
    pipeline.stop()
