"""The study's U²-Net foreground procedure for one neutral render. Run with a Python that has OpenCV."""
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

model, source, destination = sys.argv[1:]
cv2.setNumThreads(2)
net = cv2.dnn.readNetFromONNX(model)
with Image.open(source) as image:
    small = np.asarray(image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS), np.float32)
normalized = (small / 255 - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
net.setInput(normalized.transpose(2, 0, 1)[None])
prediction = net.forward(net.getUnconnectedOutLayersNames()[0])[0, 0]
if not np.isfinite(prediction).all():
    raise ValueError("Nonfinite foreground mask")
mask = cv2.resize((prediction >= .55).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
if count < 2:
    raise ValueError("No character region")
foreground = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
coverage = float(foreground.mean())
if not .07 <= coverage <= .78:
    raise ValueError(f"Unusable foreground region {coverage:.3f}")
expanded = cv2.dilate(foreground, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
soft = cv2.GaussianBlur(expanded.astype(np.float32), (31, 31), 6)
soft[soft < .025] = 0
Image.fromarray(np.round(soft * 255).astype(np.uint8)).save(destination)
print(f"{coverage:.3f}")
