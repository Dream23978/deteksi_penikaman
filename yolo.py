import cv2
from ultralytics import YOLO

# 1. Muat kedua model versi Nano (paling ringan & cepat untuk CPU laptop)
model_detect = YOLO("best.pt")           # Model custom pisau (Nano)
model_pose = YOLO("yolo11n-pose.pt")     # Model pose versi Nano (sangat cepat & enteng)

# 2. Buka sumber video dari IP Webcam HP
cap = cv2.VideoCapture(1)

# Atur buffer OpenCV agar selalu mengambil frame paling baru (anti-delay)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("YOLO Object + Pose Combined", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Object + Pose Combined", 720, 480)

# Variabel untuk frame skipping agar FPS lancar
frame_count = 0
results_detect = None
results_pose = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize gambar ke 640x480 (ringan untuk CPU, tampilan tetap jernih)
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # Jalankan AI setiap 2 frame sekali agar video tetap lancar dan responsif
    if frame_count % 2 == 0 or results_pose is None:
        results_detect = model_detect(frame, imgsz=384, conf=0.35, verbose=False)[0]
        results_pose = model_pose(frame, imgsz=384, verbose=False)[0]

    # 5. Gabungkan visualisasi orang + pose + pisau jika hasil deteksi sudah tersedia
    if results_pose is not None and results_detect is not None:
        frame_with_pose = results_pose.plot(boxes=True) 
        final_frame = results_detect.plot(img=frame_with_pose)
    else:
        final_frame = frame

    # Tampilkan hasil
    cv2.imshow("YOLO Object + Pose Combined", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
