"""
OpenVINO cihaz karsilastirmasi (kamerasiz, saf model hizi).

Ornekler:
  python benchmark_devices.py
  python benchmark_devices.py --model yolo11n_int8_openvino_model/yolo11n.xml --out results/benchmark_int8.csv
  python benchmark_devices.py --devices CPU,NPU

Olcutler:
  - Derleme suresi (compile_model, onbelleksiz)
  - Gecikme (LATENCY modu, tek istek): ortalama / p50 / p95 ms
  - Verim (THROUGHPUT modu, paralel istekler): saniyede kare (FPS)
"""
import argparse
import csv
import os
import statistics
import time

import numpy as np
import openvino as ov


def parse_args():
    p = argparse.ArgumentParser(description="OpenVINO cihaz benchmark'i")
    p.add_argument("--model", default="yolo11n_openvino_model/yolo11n.xml")
    p.add_argument("--devices", default="CPU,GPU.0,NPU", help="virgulle ayrilmis cihaz listesi")
    p.add_argument("--out", default="benchmark_results.csv", help="sonuc CSV dosyasi")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--lat-iters", type=int, default=300)
    p.add_argument("--tp-iters", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    core = ov.Core()
    model = core.read_model(args.model)
    shape = list(model.input(0).get_shape())
    x = np.random.rand(*shape).astype(np.float32)
    print(f"Model: {args.model}  | giris: {shape}")
    print(f"Mevcut cihazlar: {core.available_devices}")

    results = []
    for dev in args.devices.split(","):
        if dev not in core.available_devices:
            print(f"\n[{dev}] bulunamadi, atlaniyor.")
            continue
        name = core.get_property(dev, "FULL_DEVICE_NAME")
        print(f"\n=== {dev} ({name}) ===")

        # 1) Gecikme: tek istek, LATENCY modu
        t = time.perf_counter()
        cm = core.compile_model(model, dev, {"PERFORMANCE_HINT": "LATENCY"})
        compile_s = time.perf_counter() - t
        print(f"  Derleme: {compile_s:.2f} sn")

        req = cm.create_infer_request()
        for _ in range(args.warmup):
            req.infer({0: x})
        lat = []
        for _ in range(args.lat_iters):
            t0 = time.perf_counter()
            req.infer({0: x})
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        avg = statistics.mean(lat)
        p50 = lat[len(lat) // 2]
        p95 = lat[int(len(lat) * 0.95) - 1]
        print(f"  Gecikme: ort {avg:.2f} ms | p50 {p50:.2f} ms | p95 {p95:.2f} ms")
        del req, cm

        # 2) Verim: paralel istekler, THROUGHPUT modu
        cm_tp = core.compile_model(model, dev, {"PERFORMANCE_HINT": "THROUGHPUT"})
        nreq = cm_tp.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        queue = ov.AsyncInferQueue(cm_tp, nreq)
        for _ in range(nreq * 2):
            queue.start_async({0: x})
        queue.wait_all()
        t0 = time.perf_counter()
        for _ in range(args.tp_iters):
            queue.start_async({0: x})
        queue.wait_all()
        tp_fps = args.tp_iters / (time.perf_counter() - t0)
        print(f"  Verim:   {tp_fps:.1f} FPS  ({nreq} paralel istek)")
        del queue, cm_tp

        results.append({
            "cihaz": dev,
            "ad": name,
            "derleme_sn": round(compile_s, 2),
            "gecikme_ort_ms": round(avg, 2),
            "gecikme_p50_ms": round(p50, 2),
            "gecikme_p95_ms": round(p95, 2),
            "verim_fps": round(tp_fps, 1),
            "paralel_istek": nreq,
        })

    if not results:
        return

    print("\n" + "=" * 86)
    print(f"{'Cihaz':<7} {'Derleme(sn)':>11} {'Ort(ms)':>9} {'p50(ms)':>9} {'p95(ms)':>9} "
          f"{'Verim(FPS)':>11} {'Istek':>6}  Ad")
    print("-" * 86)
    for r in results:
        print(f"{r['cihaz']:<7} {r['derleme_sn']:>11} {r['gecikme_ort_ms']:>9} "
              f"{r['gecikme_p50_ms']:>9} {r['gecikme_p95_ms']:>9} {r['verim_fps']:>11} "
              f"{r['paralel_istek']:>6}  {r['ad']}")
    print("=" * 86)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Sonuclar {args.out} dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
