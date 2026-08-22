import cv2
import math
from pathlib import Path


# =========================
# تنظیمات
# =========================

INPUT_DIR = Path("C:/Users/win/Desktop/Input_Image")
OUTPUT_DIR = Path("C:/Users/win/Desktop/Output_Image")

MAX_PIXELS = 250_000


# =========================
# Resize بر اساس حداکثر پیکسل
# =========================

def resize_by_max_pixels(image, max_pixels):
    h, w = image.shape[:2]
    current_pixels = w * h

    if current_pixels <= max_pixels:
        return image

    scale = math.sqrt(max_pixels / current_pixels)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # تضمین اینکه از MAX_PIXELS عبور نکنیم
    while new_w * new_h > max_pixels:
        if new_w >= new_h:
            new_w -= 1
        else:
            new_h -= 1

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# =========================
# پردازش تصاویر
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

input_files = sorted(
    [
        file for file in INPUT_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ]
)

print(f"تعداد تصاویر: {len(input_files)}")
print("-" * 60)


for index, input_file in enumerate(input_files, start=1):

    image = cv2.imread(str(input_file))

    if image is None:
        print(f"[ERROR] قابل خواندن نیست: {input_file.name}")
        continue

    original_h, original_w = image.shape[:2]

    resized = resize_by_max_pixels(
        image,
        MAX_PIXELS
    )

    new_h, new_w = resized.shape[:2]

    # خروجی همیشه PNG
    output_file = OUTPUT_DIR / f"{index}.png"

    success = cv2.imwrite(
        str(output_file),
        resized,
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )

    if not success:
        print(f"[ERROR] ذخیره نشد: {output_file.name}")
        continue

    original_pixels = original_w * original_h
    new_pixels = new_w * new_h

    print(
        f"{index}. {input_file.name} | "
        f"{original_w}x{original_h} ({original_pixels:,} px)"
        f" → "
        f"{new_w}x{new_h} ({new_pixels:,} px)"
        f" | PNG"
    )


print("-" * 60)
print("finished")
