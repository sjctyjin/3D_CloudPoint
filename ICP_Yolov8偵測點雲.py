import numpy as np
import open3d as o3d
import cv2
import copy
import os
from ultralytics import YOLO  # 確保安裝了ultralytics庫


def extract_object_pointcloud(rgb_image, depth_image, camera_intrinsic, detection_box, depth_scale=1000.0):
    """
    從RGB-D數據中提取目標物體的點雲

    Args:
        rgb_image: RGB圖像 (HxWx3, np.uint8)
        depth_image: 深度圖像 (HxW, np.float or np.uint16)
        camera_intrinsic: 相機內參矩陣 (3x3)
        detection_box: YOLOv8檢測結果，格式[x1, y1, x2, y2]
        depth_scale: 深度縮放因子，例如Intel D435通常為1000(mm)

    Returns:
        object_pointcloud: 目標物體的點雲(Open3D PointCloud)
    """
    # 提取邊界框座標
    x1, y1, x2, y2 = map(int, detection_box)

    # 確保邊界框在圖像範圍內
    H, W = depth_image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    # 創建空點雲
    object_pointcloud = o3d.geometry.PointCloud()
    points = []
    colors = []

    # 相機內參數
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]

    # 從邊界框區域提取點雲
    for v in range(y1, y2):
        for u in range(x1, x2):
            # 獲取深度值，跳過無效值
            z = depth_image[v, u]
            if z <= 0 or np.isnan(z):
                continue

            # 將深度值轉換為米
            z = z / depth_scale

            # 反投影到3D空間
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # 添加點和顏色
            points.append([x, y, z])
            colors.append(rgb_image[v, u] / 255.0)  # 將顏色歸一化到[0,1]

    # 如果沒有有效點，返回空點雲
    if not points:
        print("警告：邊界框內沒有有效深度點")
        return None

    # 設置點雲數據
    object_pointcloud.points = o3d.utility.Vector3dVector(np.array(points))
    object_pointcloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    # 可選：執行點雲處理，如濾除噪點
    object_pointcloud = object_pointcloud.voxel_down_sample(voxel_size=0.005)  # 5mm體素化以減少點數

    # 可選：估計法線
    object_pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    return object_pointcloud


def detect_object_with_yolo(image, depth_image, camera_intrinsic, model_path, confidence=0.2, target_class=None, depth_scale=1000.0):
    """
    使用YOLOv8檢測圖像中的目標物體

    Args:
        image: 輸入圖像
        model_path: YOLOv8模型路徑
        confidence: 檢測置信度閾值
        target_class: 目標類別名稱(如'cup', 'bottle'等)，None表示檢測分數最高的物體

    Returns:
        box: 檢測框 [x1, y1, x2, y2]
        class_name: 類別名稱
        conf: 置信度
    """
    # 加載YOLO模型
    model = YOLO(model_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 轉換為OpenCV格式
    # 運行檢測
    results = model(image)[0]

    boxes = results.boxes.cpu().numpy()
    # 相機內參數
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]

    X = None
    Y = None
    Z = None
    # 遍歷每個檢測框
    for i, box in enumerate(boxes):
        # 獲取邊界框座標
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 獲取類別和置信度
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = results.names[cls_id]

        # 在框中心獲取深度值
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # 獲取中心點的深度值
        depth_value = depth_image[center_y, center_x]
        if depth_value <= 0 or np.isnan(depth_value):
            depth_meter = 0.0
        else:
            depth_meter = depth_value / depth_scale  # 轉換為米

        # 計算3D坐標
        X = (center_x - cx) * depth_meter / fx
        Y = (center_y - cy) * depth_meter / fy
        Z = depth_meter
        # 在圖像上畫出邊界框
        color = (0, 255, 0)  # 綠色框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 添加類別和置信度標籤
        label = f"{cls_name} {conf:.2f} "
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"檢測到 {cls_name}，置信度: {conf:.2f}")



    # 找到目標物體
    best_box = None
    best_class = None
    best_conf = 0

    for box in results.boxes:
        # 獲取檢測結果
        class_id = int(box.cls)
        conf = float(box.conf)
        class_name = results.names[class_id]
        print("辨識度 : ", conf)
        print("辨識度基準 : ", best_conf)
        # 檢查置信度
        if conf < confidence:
            continue

        # 如果指定了目標類別，則只保留該類別
        if target_class is not None and class_name != target_class:
            continue

        # 找到置信度最高的物體
        if conf > best_conf:
            best_conf = conf
            best_class = class_name
            best_box = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

    # cv2.imshow("YOLOv8", image)
    # cv2.waitKey(0)

    if best_box is not None:
        return best_box, best_class, best_conf,X,Y,Z
    else:
        return None, None, None


def main():
    # 加載RGB和深度圖像
    rgb_image_path = "rgb_image.png"  # 替換為您的RGB圖像
    depth_image_path = "depth_image.png"  # 替換為您的深度圖像

    # 檢查文件是否存在
    if not os.path.exists(rgb_image_path) or not os.path.exists(depth_image_path):
        print(f"錯誤: 找不到圖像文件")
        return

    # 讀取圖像
    rgb_image = cv2.imread(rgb_image_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # 轉換為RGB
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # 讀取原始深度值

    # 相機內參 (根據您的D435相機實際參數設置)
    # 注意：這些值應該從相機標定中獲取
    # 相機內參矩陣:
    # [[433.56628418   0.         327.215271]
    #  [0.         432.98043823 244.18916321]
    # [0.
    # 0.
    # 1.]]
    camera_intrinsic = np.array([
        [433.56628418, 0, 327.215271],
        [0, 432.98043823, 244.18916321],
        [0, 0, 1]
    ])

    # 使用YOLOv8檢測物體
    yolo_model_path = "pamu.pt"  # 使用預訓練模型或您的自定義模型
    target_class = "pami"  # 想要檢測的目標類別，設置為None檢測任何物體

    box, class_name, conf,X,Y,Z = detect_object_with_yolo(rgb_image,depth_image,camera_intrinsic, yolo_model_path, target_class=target_class)
    print(f"YOLOv8檢測結果: {box}, {class_name}, {conf}")
    if box is None:
        print("未檢測到目標物體")
        return

    print(f"檢測到物體: {class_name}，置信度: {conf:.2f}，邊界框: {box}")

    # 可視化檢測結果
    vis_image = rgb_image.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis_image, f"{class_name}: {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(vis_image, (int((x1+x2)/2), int((y1+y2)/2)), 5, (0, 255, 0), -1)
    position_text = f"3D: ({X:.3f}, {Y:.3f}, {Z:.3f})m"
    cv2.putText(vis_image, position_text, (x1, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 顯示檢測結果
    cv2.imshow("Object Detection", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

    # 提取目標物體的點雲
    object_pointcloud = extract_object_pointcloud(rgb_image, depth_image, camera_intrinsic, box)
    # 注意：請使用您實際相機的內參值替換這些值
    intrinsic = np.array([
        [433.56628418, 0, 327.215271],
        [0, 432.98043823, 244.18916321],
        [0, 0, 1]
    ])


    if object_pointcloud is None:
        print("無法提取有效的物體點雲")
        return

    # 計算抓取姿態
    # 這裡可以使用您之前的抓取姿態計算函數
    from ICP_姿態轉換_Open3D相機視角 import visualize_with_grasp  # 導入您的抓取計算模塊

    # 顯示提取的點雲
    print("顯示目標物體點雲...")
    # o3d.visualization.draw_geometries([object_pointcloud],
    #                                   window_name="目標物體點雲",
    #                                   width=800, height=600)
    cam_parm = {
        "position":  np.array([X, Y, Z]),
        "rotation": np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]),
    }
    # 計算抓取姿態
    visualize_with_grasp(object_pointcloud,False,cam_parm)
    # visualize_with_grasp(object_pointcloud)


if __name__ == "__main__":
    main()