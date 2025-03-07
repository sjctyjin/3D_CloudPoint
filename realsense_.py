import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time

# 初始化YOLOv8模型
model = YOLO("pamu.pt")  # 載入您的自訂權重檔

# 配置RealSense管線
pipeline = rs.pipeline()
config = rs.config()

# 啟用彩色和深度串流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 啟動管線
profile = pipeline.start(config)

# 獲取相機內參
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# 建立相機內參矩陣
camera_intrinsic = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
])

print("相機內參矩陣:")
print(camera_intrinsic)

# 創建對齊物件來對齊深度幀與彩色幀
align = rs.align(rs.stream.color)

# 創建深度圖的彩色映射器
colorizer = rs.colorizer()

# 等待幾幀以穩定相機
for _ in range(30):
    pipeline.wait_for_frames()

try:
    while True:
        start_time = time.time()

        # 等待一組完整的幀
        frames = pipeline.wait_for_frames()

        # 對齊幀
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 轉換為numpy陣列
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 執行YOLOv8檢測
        results = model(color_image)

        # 處理檢測結果
        for result in results:
            boxes = result.boxes.cpu().numpy()

            # 遍歷每個檢測框
            for i, box in enumerate(boxes):
                # 獲取邊界框座標
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 獲取類別和置信度
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]

                # 在框中心獲取深度值
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                depth_value = depth_frame.get_distance(center_x, center_y)

                # 反投影到3D空間（相機座標系）
                x = (center_x - color_intrinsics.ppx) * depth_value / color_intrinsics.fx
                y = (center_y - color_intrinsics.ppy) * depth_value / color_intrinsics.fy
                z = depth_value

                # 在圖像上畫出邊界框
                color = (0, 255, 0)  # 綠色框
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)

                # 添加類別和置信度標籤
                label = f"{cls_name} {conf:.2f} Depth: {depth_value:.2f}m"
                cv2.putText(color_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 打印3D座標
                position_text = f"3D: ({x:.2f}, {y:.2f}, {z:.2f})m"
                cv2.putText(color_image, position_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                print(f"檢測到 {cls_name}，置信度: {conf:.2f}, 位置: ({x:.2f}, {y:.2f}, {z:.2f})m")

        # 計算每秒幀數
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 創建深度圖的彩色表示
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # 並排顯示圖像
        images = np.hstack((color_image, depth_colormap))

        # 顯示結果
        cv2.namedWindow('RealSense + YOLOv8', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense + YOLOv8', images)

        # 按'q'鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"發生錯誤: {e}")
finally:
    # 停止管線
    pipeline.stop()
    cv2.destroyAllWindows()