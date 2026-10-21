"""
OpenVINO + YOLO11n ile gercek zamanli webcam nesne tespiti.

Ornekler:
  python yolo_webcam.py
  python yolo_webcam.py --device NPU --camera 1
  python yolo_webcam.py --model yolo11n_int8_openvino_model/yolo11n.xml --device GPU.0

Cikmak icin: q veya ESC
"""
import argparse
import sys
import time

import cv2
import numpy as np
import openvino as ov

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def parse_args():
    p = argparse.ArgumentParser(description="OpenVINO YOLO11n webcam demo")
    p.add_argument("--model", default="yolo11n_openvino_model/yolo11n.xml",
                   help="OpenVINO IR modeli (.xml)")
    p.add_argument("--device", default="GPU.0",
                   help="CPU, GPU.0, NPU veya AUTO (varsayilan: GPU.0)")
    p.add_argument("--camera", type=int, default=0, help="kamera index'i (varsayilan: 0)")
    p.add_argument("--backend", default="MSMF", choices=["MSMF", "DSHOW"],
                   help="Windows kamera backend'i (varsayilan: MSMF)")
    p.add_argument("--conf", type=float, default=0.40, help="guven esigi (varsayilan: 0.40)")
    p.add_argument("--iou", type=float, default=0.50, help="NMS IoU esigi (varsayilan: 0.50)")
    p.add_argument("--no-cache", action="store_true",
                   help="model onbellegini kapat (ilk derleme suresini olcmek icin)")
    return p.parse_args()


def main():
    args = parse_args()
    size = 640
    colors = np.random.default_rng(42).integers(60, 255, size=(len(COCO), 3)).tolist()

    core = ov.Core()
    if not args.no_cache:
        # Derlenmis modeli diske yazar; ikinci acilistan itibaren derleme cok hizlanir
        core.set_property({"CACHE_DIR": "ov_cache"})
    try:
        dev_name = core.get_property(args.device, "FULL_DEVICE_NAME")
    except Exception:
        dev_name = args.device
    print(f"Model: {args.model}")
    print(f"Derleniyor -> {args.device} ({dev_name}) ...")
    t = time.perf_counter()
    compiled = core.compile_model(args.model, args.device)
    print(f"Derleme tamam: {time.perf_counter() - t:.1f} sn")
    output = compiled.output(0)

    api = cv2.CAP_DSHOW if args.backend == "DSHOW" else cv2.CAP_MSMF
    cap = cv2.VideoCapture(args.camera, api)
    print(f"Kamera: index {args.camera}, backend {args.backend}")
    if not cap.isOpened():
        print("Kamera acilamadi. 'python list_cameras.py' ile dogru index'i bulun.")
        sys.exit(1)

    fps, infer_ms_avg = 0.0, 0.0
    while True:
        loop_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("Kameradan goruntu alinamadi.")
            break
        h, w = frame.shape[:2]

        # On isleme: 640x640, RGB, NCHW, 0-1 araligi
        img = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
        blob = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        # Cikarim
        t0 = time.perf_counter()
        pred = compiled(blob)[output][0]          # (84, 8400)
        infer_ms = (time.perf_counter() - t0) * 1000

        # Son isleme: kutular + sinif skorlari + NMS
        pred = pred.T                              # (8400, 84)
        scores = pred[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]
        keep = confs > args.conf
        pred, cls_ids, confs = pred[keep], cls_ids[keep], confs[keep]

        sx, sy = w / size, h / size
        boxes = [
            [int((cx - bw / 2) * sx), int((cy - bh / 2) * sy), int(bw * sx), int(bh * sy)]
            for cx, cy, bw, bh in pred[:, :4]
        ]
        idxs = cv2.dnn.NMSBoxes(boxes, confs.tolist(), args.conf, args.iou)

        for i in np.array(idxs).flatten().astype(int):
            x, y, bw, bh = boxes[i]
            c = int(cls_ids[i])
            label = f"{COCO[c]} {confs[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colors[c], 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), colors[c], -1)
            cv2.putText(frame, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # FPS / sure (yumusatilmis)
        loop_fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
        fps = loop_fps if fps == 0 else 0.9 * fps + 0.1 * loop_fps
        infer_ms_avg = infer_ms if infer_ms_avg == 0 else 0.9 * infer_ms_avg + 0.1 * infer_ms

        info = f"{args.device} | FPS: {fps:.1f} | Inference: {infer_ms_avg:.1f} ms"
        cv2.rectangle(frame, (0, 0), (len(info) * 11 + 10, 32), (0, 0, 0), -1)
        cv2.putText(frame, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("OpenVINO YOLO11n", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Ortalama -> {args.device}: {fps:.1f} FPS, {infer_ms_avg:.1f} ms/kare")


if __name__ == "__main__":
    main()
