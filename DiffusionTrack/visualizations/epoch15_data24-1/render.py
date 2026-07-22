from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CLASSES = ["car", "bike", "pedestrian", "van", "truck", "bus", "tricycle", "awning-bike"]
COLORS = [
    (50, 220, 255), (255, 150, 40), (255, 80, 180), (70, 220, 70),
    (80, 80, 255), (220, 100, 60), (200, 200, 60), (180, 80, 220),
]


def read_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            rows.append([float(x) for x in line.split(",")])
    return rows


def pseudo_rgb(array):
    # First three MSI bands, independently contrast-stretched for display only.
    image = array[..., [2, 1, 0]].astype(np.float32)
    for channel in range(3):
        lo, hi = np.percentile(image[..., channel], (1, 99))
        image[..., channel] = np.clip((image[..., channel] - lo) * 255.0 / max(hi - lo, 1), 0, 255)
    return image.astype(np.uint8)


def panel(base, title, rows, is_prediction=False):
    canvas = base.copy()
    for row in rows:
        points = np.rint(row[2:10]).astype(np.int32).reshape(4, 2)
        cls = int(row[11])
        color = COLORS[cls % len(COLORS)] if is_prediction else (30, 255, 30)
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)
        if is_prediction:
            score = row[10]
            label = f"{CLASSES[cls] if cls < len(CLASSES) else cls} {score:.2f}"
            x, y = points[0]
            cv2.putText(canvas, label, (max(0, x), max(16, y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (15, 15, 15), -1)
    cv2.putText(canvas, title, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


gt = read_rows(ROOT / "ground_truth.txt")
pred = read_rows(ROOT / "predictions.txt")

for npy_path in sorted(ROOT.glob("*.npy")):
    frame = int(npy_path.stem)
    base = pseudo_rgb(np.load(npy_path))
    frame_gt = [row for row in gt if int(row[0]) == frame]
    frame_pred = [row for row in pred if int(row[0]) == frame]
    ranked = sorted(frame_pred, key=lambda row: row[10], reverse=True)
    top_n = ranked[:len(frame_gt)]
    normal = [row for row in ranked if row[10] >= 0.4]

    panels = [
        panel(base, f"Frame {frame:02d} | MSI pseudo-RGB (bands 0/1/2)", []),
        panel(base, f"Ground truth | {len(frame_gt)} objects", frame_gt),
        panel(base, f"Epoch 15 | top {len(top_n)} of {len(frame_pred)} predictions", top_n, True),
        panel(base, f"Epoch 15 | score >= 0.4 | {len(normal)} predictions", normal, True),
    ]
    overview = np.concatenate(panels, axis=1)
    overview = cv2.resize(overview, (overview.shape[1] // 2, overview.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(ROOT / f"frame_{frame:02d}_comparison.jpg"), overview,
                [cv2.IMWRITE_JPEG_QUALITY, 94])
