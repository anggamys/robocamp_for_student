import os
import cv2
from ultralytics import YOLO

# ---------- Konfigurasi ----------
MODEL_PATH = os.path.join("model_asv_2024", "bola.pt")
IMAGE_PATH = os.path.join("assets", "sample.jpg")
OUTPUT_PATH = os.path.join("assets", "output.jpg")
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
    
    # Buka Gambar
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Gambar tidak ditemukan: {IMAGE_PATH}")
        return
        
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("[ERROR] Tidak dapat membuka gambar.")
        return

    # Inference YOLO
    results = model(frame, conf=CONF_THRESH, verbose=False)

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

    # Simpan hasil deteksi
    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"Hasil deteksi disimpan ke: {OUTPUT_PATH}")

    # Tampilkan gambar
    cv2.imshow("Deteksi Bola Hijau & Biru - Image", frame)
    print("Tekan tombol apa saja pada jendela gambar untuk keluar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()