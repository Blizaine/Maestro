"""Auto-crop faces from a character sheet into individual images.

Detects faces (frontal + profile, multi-scale) and saves each as a padded
head-and-shoulders crop for use as individual character references.
One-off helper — not part of the launcher.
"""
import cv2
from pathlib import Path

SHEET = r"Z:\VelaSong\YouTubeVideo\Assets\BurnTheBlacktop\CharacterSheet.jpg"
OUT = Path(r"Z:\VelaSong\YouTubeVideo\Assets\BurnTheBlacktop\characters")
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(SHEET)
if img is None:
    raise SystemExit(f"Could not read {SHEET}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape[:2]

frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

detections = []

# Frontal, multi-scale (catches both large headshots and small body-shot faces)
for scale in (1.05, 1.1, 1.2):
    for (x, y, w, h) in frontal.detectMultiScale(gray, scaleFactor=scale, minNeighbors=4, minSize=(28, 28)):
        detections.append((x, y, w, h))

# Profile (left-facing) + flipped (right-facing)
for (x, y, w, h) in profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28)):
    detections.append((x, y, w, h))
flipped = cv2.flip(gray, 1)
for (x, y, w, h) in profile.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28)):
    detections.append((W - (x + w), y, w, h))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# Non-max suppression on overlapping detections
keep = []
for d in sorted(detections, key=lambda d: d[2] * d[3], reverse=True):
    if all(iou(d, k) < 0.25 for k in keep):
        keep.append(d)

# Sort top-to-bottom then left-to-right for stable numbering
keep = sorted(keep, key=lambda d: (round(d[1] // 150), d[0]))

for i, (x, y, w, h) in enumerate(keep, 1):
    cx, cy = x + w / 2, y + h / 2
    # Head-and-shoulders: generous above (hair), moderate below (shoulders)
    half_w = w * 1.3
    top = cy - h * 1.7
    bot = cy + h * 1.6
    x0, x1 = int(cx - half_w), int(cx + half_w)
    y0, y1 = int(top), int(bot)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    out = OUT / f"face_{i:02d}.jpg"
    cv2.imwrite(str(out), crop)
    print(f"{out.name}: face box ({x},{y},{w}x{h}) -> crop {crop.shape[1]}x{crop.shape[0]}")

print(f"\nSaved {len(keep)} crops to {OUT}")
