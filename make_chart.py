"""
results/ altindaki CSV'lerden README grafigini uretir -> assets/benchmark.png
Kullanim: python make_chart.py   (matplotlib gerekir: pip install matplotlib)
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

FP32_COLOR = "#2a78d6"
INT8_COLOR = "#eb6834"
TEXT = "#2b2b29"
MUTED = "#6b6a63"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["cihaz"]: r for r in csv.DictReader(f)}


fp32 = load("results/benchmark_fp32.csv")
int8 = load("results/benchmark_int8.csv")
devices = ["CPU", "GPU.0", "NPU"]
labels = ["CPU", "iGPU", "NPU"]

panels = [
    ("Gecikme (ms) — düşük daha iyi", "gecikme_ort_ms", "{:.1f}"),
    ("Verim (FPS) — yüksek daha iyi", "verim_fps", "{:.0f}"),
]

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
x = np.arange(len(devices))
width = 0.36

for ax, (title, key, fmt) in zip(axes, panels):
    a = [float(fp32[d][key]) for d in devices]
    b = [float(int8[d][key]) for d in devices]
    bars_a = ax.bar(x - width / 2 - 0.01, a, width, color=FP32_COLOR, label="FP32", zorder=3)
    bars_b = ax.bar(x + width / 2 + 0.01, b, width, color=INT8_COLOR, label="INT8", zorder=3)
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.annotate(fmt.format(bar.get_height()),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, color=TEXT)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=12)
    ax.set_xticks(x, labels, color=TEXT)
    ax.tick_params(axis="y", colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(0, max(a + b) * 1.15)

handles, leg_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc="upper right", ncol=2, frameon=False,
           bbox_to_anchor=(0.99, 1.0), labelcolor=TEXT)
fig.suptitle("YOLO11n · OpenVINO 2026.4 · Intel Core Ultra 9 185H",
             x=0.01, ha="left", fontsize=13, color=TEXT, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("assets/benchmark.png", dpi=160, facecolor=SURFACE)
print("assets/benchmark.png kaydedildi.")
