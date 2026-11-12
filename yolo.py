import cv2
import time
from ultralytics import YOLO

# ─────────────────────────────────────────
#  1. MUAT MODEL
# ─────────────────────────────────────────
model_detect = YOLO("best.pt")           # Model custom (hand & knife)
model_pose   = YOLO("yolo11n-pose.pt")  # Model pose skeleton

print("Kelas model deteksi:", model_detect.names)

# ─────────────────────────────────────────
#  2. BUKA KAMERA
# ─────────────────────────────────────────
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Security Cam - YOLO + Pose + ByteTrack", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Security Cam - YOLO + Pose + ByteTrack", 720, 480)

# ─────────────────────────────────────────
#  3. STATUS & COOLDOWN
# ─────────────────────────────────────────
# Menyimpan waktu terakhir deteksi pisau per ID objek
last_seen    = {}   # {track_id: timestamp terakhir terdeteksi}
knife_ids    = set()  # kumpulan track_id yang teridentifikasi sebagai pisau

KNIFE_CLASS_ID = 1   # Sesuai model: {0:'hand', 1:'knife'}

# ─────────────────────────────────────────
#  4. LOOP UTAMA
# ─────────────────────────────────────────
frame_count    = 0
results_pose   = None
results_detect = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # ── Deteksi Pisau pakai ByteTrack (setiap frame untuk tracking mulus) ──
    # .track() = model_detect + ByteTrack otomatis
    # persist=True → tracker mengingat ID antar frame walaupun objek hilang sesaat
    track_results = model_detect.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.36,
        # Tanpa filter classes → deteksi SEMUA kelas: hand (0) + knife (1)
        verbose=False,
    )[0]

    # ── Model Pose setiap 2 frame agar FPS tetap tinggi ──
    if frame_count % 2 == 0 or results_pose is None:
        results_pose = model_pose(frame, imgsz=384, verbose=False)[0]

    # ─────────────────────────────────────────
    #  5. BACA TRACK ID & TAMPILKAN INFO
    # ─────────────────────────────────────────
    now        = time.time()
    knife_now  = set()   # ID pisau yang terdeteksi di frame ini
    hand_now   = set()   # ID tangan yang terdeteksi di frame ini

    if track_results.boxes is not None and len(track_results.boxes) > 0:
        for box in track_results.boxes:
            if box.id is None:
                continue

            track_id = int(box.id[0])
            cls_id   = int(box.cls[0])

            if cls_id == KNIFE_CLASS_ID:
                knife_now.add(track_id)
                last_seen[track_id] = now
                knife_ids.add(track_id)
            else:  # cls_id == 0 → hand
                hand_now.add(track_id)

    # Hitung jumlah orang dari model pose (jumlah bounding box person yang terdeteksi)
    person_count = len(results_pose.boxes) if results_pose is not None and results_pose.boxes is not None else 0

    # ─────────────────────────────────────────
    #  6. RENDER FRAME
    # ─────────────────────────────────────────
    # Gambar skeleton pose dulu
    if results_pose is not None:
        base_frame = results_pose.plot(img=frame.copy(), boxes=False)
    else:
        base_frame = frame.copy()

    # Gambar kotak tracking pisau di atasnya
    final_frame = track_results.plot(img=base_frame)

    # ── Info panel status ──
    total_knives = len(knife_now)
    total_hands  = len(hand_now)

    # Warna panel berubah merah jika ada pisau
    panel_color = (0, 0, 180) if total_knives > 0 else (30, 30, 30)
    cv2.rectangle(final_frame, (5, 5), (300, 90), panel_color, -1)

    # Baris 1: Status utama
    if total_knives > 0:
        cv2.putText(final_frame, f"! PISAU TERDETEKSI: {total_knives}",
                    (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 80, 255), 2)
    else:
        cv2.putText(final_frame, "STATUS: AMAN",
                    (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 0), 2)

    # Baris 2: Jumlah orang
    cv2.putText(final_frame, f"Orang   : {person_count}",
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Baris 3: Jumlah tangan dan pisau
    cv2.putText(final_frame, f"Tangan  : {total_hands}   |   Pisau: {total_knives}",
                (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    # ─────────────────────────────────────────
    #  7. TAMPILKAN JENDELA
    # ─────────────────────────────────────────
    cv2.imshow("Security Cam - YOLO + Pose + ByteTrack", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nSesi selesai. Total ID pisau pernah muncul:", knife_ids)
