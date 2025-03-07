## ICP_Yolov8偵測點雲.py : 
調用pamu.pt偵測rgb_image.png中的物體，找到偵測物後將其點雲分割，並調用ICP_姿態轉換_Open3D相機視角.py 計算目標物抓取最佳姿態。

## ICP_生成抓取姿態.py : 
讀取bun000.ply點雲圖後，計算點雲相對於原點的座標姿態及位移，並在XY平面上找到點雲中相對位置最窄的地方 生成夾爪位置。

## ICP_姿態轉換_Open3D相機視角.py : 
預設啟動會讀取output_point_cloud.ply點雲，並在Open3D中旋轉到固定視角後計算與該視角為原點的相對座標姿態位移，並在XY平面上找到點雲中相對位置最窄的地方 生成夾爪位置。

## realsense_.py : 
開啟Realsense相機，並調用yolov8 pamu.pt模型做影像偵測

## realsense_open3d.py : 
調用資料夾中的 rgb_image.png 與 depth_image.png並顯示在open3D上，且顯示相機位置

## 保存深度圖(6DOF用).py : 
開啟Realsense相機並直接擷取一張圖，並保存成三個檔案 : rgb_image.png、 depth_image.png 與 depth_image_16bit.png

## 深度圖轉換ply點雲.py : 
執行過"保存深度圖(6DOF用).py" 得到rgb_image.png 與 depth_image.png兩張圖片後，轉出output_point_cloud.ply檔案 提供"ICP_姿態轉換_Open3D相機視角.py " 執行範例使用。