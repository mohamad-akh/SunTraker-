import cv2
import glob
import os
from core.sun_tracker import SunTracker

# پوشه تصاویر
IMAGE_FOLDER = "images"

# محل ذخیره ویدئوی موقت (داخل پروژه)
VIDEO_FILE = "temp_video.avi"

# پسوندهای مجاز
extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

images = []
for ext in extensions:
    images.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))

images = sorted(images)

if not images:
    raise Exception("هیچ تصویری داخل پوشه images پیدا نشد.")

# اولین تصویر
first = cv2.imread(images[0])
height, width = first.shape[:2]

# ساخت ویدئو
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(VIDEO_FILE, fourcc, 25, (width, height))

for img_path in images:
    frame = cv2.imread(img_path)

    if frame is None:
        continue

    # اگر اندازه تصاویر متفاوت بود
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))

    writer.write(frame)

writer.release()

print("Video ساخته شد:", VIDEO_FILE)

# اجرای SunTracker بدون هیچ تغییری
tracker = SunTracker(video_source=VIDEO_FILE)
tracker.run()