"""
Bilgisayardaki kameralari tarar; hangi index/backend gercek goruntu veriyor gosterir.
Kullanim: python list_cameras.py
"""
import time

import cv2

BACKENDS = {"MSMF": cv2.CAP_MSMF, "DSHOW": cv2.CAP_DSHOW}

print("Kameralar taraniyor (birkac saniye surebilir)...\n")
for idx in range(5):
    for name, api in BACKENDS.items():
        cap = cv2.VideoCapture(idx, api)
        if not cap.isOpened():
            continue
        frame = None
        t0 = time.perf_counter()
        for _ in range(10):              # ilk kareler bos gelebilir, birkac tane oku
            ok, f = cap.read()
            if ok:
                frame = f
        elapsed = time.perf_counter() - t0
        cap.release()
        if frame is None:
            print(f"index {idx} | {name:5} | acildi ama kare gelmedi")
            continue
        h, w = frame.shape[:2]
        brightness = frame.mean()
        fps = 10 / elapsed if elapsed > 0 else 0
        status = "SIYAH / BOS" if brightness < 5 else "GORUNTU VAR"
        print(f"index {idx} | {name:5} | {w}x{h} | parlaklik {brightness:5.1f} | ~{fps:4.1f} FPS | {status}")

print("\n'GORUNTU VAR' olan satiri kullan:")
print("  python yolo_webcam.py --camera <index> --backend <backend>")
