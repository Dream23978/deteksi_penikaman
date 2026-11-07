import cv2
from ultralytics import YOLO

# 1. Muat kedua model (Bisa menggunakan YOLOv8, YOLO11, atau YOLOv12 jika tersedia)
model_detect = YOLO("yolo11n.pt")       # Untuk mendeteksi objek umum (kursi, laptop, dll.)
model_pose = YOLO("yolo11n-pose.pt")   # Untuk mendeteksi orang + titik kunci tubuh

# 2. Buka sumber video (0 untuk webcam, atau masukkan path file video)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Jalankan deteksi objek umum
    results_detect = model_detect(frame, verbose=False)[0]
    
    # 4. Jalankan estimasi pose (hanya deteksi manusia + pose)
    results_pose = model_pose(frame, verbose=False)[0]

    # 5. Gabungkan visualisasi hasil ke frame asli
    # Ambil frame yang sudah digambari kotak objek umum
    annotated_frame = results_detect.plot() 
    
    # Ambil visualisasi pose (kerangka tubuh), lalu overlay ke frame tadi
    # Menggunakan conf=False/boxes=False jika Anda ingin menyembunyikan kotak duplikat manusia
    pose_frame = results_pose.plot(boxes=False) 
    
    # Gabungkan gambar (menggunakan OpenCV untuk menumpuk hasilnya)
    final_frame = cv2.addWeighted(annotated_frame, 0.5, pose_frame, 0.5, 0)

    # Tampilkan hasil penggabungan
    cv2.imshow("YOLO Object + Pose Combined", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
