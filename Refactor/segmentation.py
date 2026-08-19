import os
import cv2
import numpy as np
import time

folder_path = r"C:/Users/win/Desktop/SunTrackerSplit/SunTracker-Refactor/Refactor/Images"
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

if not os.path.exists(folder_path):
    print(f"Error: Folder path does not exist -> {folder_path}")
else:
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(image_extensions):
            continue

        img_path = os.path.join(folder_path, file_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        # ۱. تغییر سایز
        # resized_img = cv2.resize(image, (640, 480))
        resized_img = image


        # --- شروع اندازه‌گیری زمان پردازش ---
        start_time = time.perf_counter()

        # ۲. تبدیل به LAB و گرفتن کانال L
        lab_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
        l_channel = lab_img[:, :, 0]

        # ۳. فیلتر گوسی
        blurred_l = cv2.GaussianBlur(l_channel, (21, 21), 0)

        # ۴. پیدا کردن بیشینه روشنایی (مرکز خورشید)
        _, max_val, _, seed_point = cv2.minMaxLoc(blurred_l)

        # ۵. ساخت ماسک کمکی
        h, w = blurred_l.shape
        mask_temp = np.zeros((h + 2, w + 2), dtype=np.uint8)

        LO_DIFF = 3
        UP_DIFF = 0

        flags = 4 | (
            255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE

        # اجرای سگمنتیشن
        cv2.floodFill(
            blurred_l.copy(),
            mask_temp,
            seed_point,
            0,
            LO_DIFF,
            UP_DIFF,
            flags,
        )

        sun_mask = mask_temp[1: h + 1, 1: w + 1]

        # --- پایان اندازه‌گیری زمان پردازش ---
        end_time = time.perf_counter()

        # محاسبه زمان به میلی‌ثانیه و فریم بر ثانیه
        execution_time_ms = (end_time - start_time) * 1000
        fps = 1000 / execution_time_ms if execution_time_ms > 0 else 0

        print(
            f"Image: {file_name} | Processing Time: {execution_time_ms:.2f} ms | FPS: {fps:.1f}")

        # ۶. مشخص کردن نقطه مرکز روی تصویر اصلی
        visual_img = resized_img.copy()
        cv2.circle(visual_img, seed_point, 5, (0, 0, 255), -1)

        # نمایش زمان روی تصویر
        time_text = f"Time: {execution_time_ms:.2f} ms ({fps:.1f} FPS)"
        cv2.putText(
            visual_img,
            time_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # ۷. تبدیل ماسک به ۳ کاناله برای نمایش کنار هم
        mask_3channel = cv2.cvtColor(sun_mask, cv2.COLOR_GRAY2BGR)

        # چسباندن تصویر اصلی و ماسک
        combined_view = np.hstack((visual_img, mask_3channel))

        # ۸. نمایش پنجره
        cv2.imshow("Original (Left) vs Segmented Mask (Right)", combined_view)

        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
