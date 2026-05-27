import os
import cv2
from ultralytics import YOLO

# ---------- Konfigurasi ----------
MODEL_PATH = os.path.join("model_asv_2024", "bola.pt")
VIDEO_PATH = os.path.join("video", "edit_perjalanan_kapal.mp4")
CONF_THRESH = 0.35

# Warna dalam format BGR (Blue, Green, Red)
COLOR_GREEN = (0, 255, 0)   # Hijau
COLOR_BLUE  = (255, 0, 0)   # Biru

def main():
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
        return
        
    model = YOLO(MODEL_PATH)
    
    # Buka Video
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video tidak ditemukan: {VIDEO_PATH}")
        return
        
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka video.")
        return

    print("Memproses video... Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai.")
            break

        # Inference YOLO pada frame
        results = model(frame, conf=CONF_THRESH, stream=True, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Klasifikasi berdasarkan class ID
                # Class 0: Green (Hijau), Class 1: Red (direpresentasikan sebagai Biru)
                if cls_id == 0:
                    color = COLOR_GREEN
                    label = f"Bola Hijau {conf:.0%}"
                elif cls_id == 1:
                    color = COLOR_BLUE
                    label = f"Bola Biru {conf:.0%}"
                else:
                    continue

                # Gambar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Gambar label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Tampilkan frame
        cv2.imshow("Deteksi Bola Hijau & Biru - Video", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")

if __name__ == "__main__":
    main()
