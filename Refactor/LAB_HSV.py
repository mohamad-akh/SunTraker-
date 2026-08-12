import cv2
import numpy as np
import sys
import os


def process_single_frame(frame, title_prefix=""):
    # استخراج کانال‌های BGR (توجه: OpenCV به‌صورت BGR خوانده می‌شود)
    b_chan, g_chan, r_chan = cv2.split(frame)

    # تبدیل به HSV و LAB
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_lab_chan = cv2.split(lab)

    # تعریف تمامی کانال‌ها به همراه رنگ متناظر جهت نمایش در پنل (BGR)
    channels = {
        # RGB Channels
        "Red (R)": (r_chan, (0, 0, 255)),          # قرمز
        "Green (G)": (g_chan, (0, 255, 0)),        # سبز
        "Blue (B)": (b_chan, (255, 0, 0)),         # آبی
        # HSV
        "Hue (H)": (h_chan, (255, 100, 0)),        # آبی‌فیروزه‌ای
        "Sat (S)": (s_chan, (0, 255, 128)),        # سبز روشن
        "Val (V)": (v_chan, (128, 0, 255)),        # بنفش‌قرمز
        # LAB
        "Light (L)": (l_chan, (255, 255, 0)),      # فیروزه‌ای
        "a* (G-R)": (a_chan, (255, 0, 255)),       # ارغوانی
        "b* (B-Y)": (b_lab_chan, (0, 255, 255))    # زرد
    }

    marked_frame = frame.copy()
    h, w = frame.shape[:2]

    # پنل مشکی در سمت راست تصویر
    panel_width = 380
    info_panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    print("=" * 65)
    print(f"📊 آمار حداقل و حداکثر کانال‌ها [{title_prefix}]:")
    print("=" * 65)

    y_offset = 25
    line_height = 20

    # تیتر پنل
    cv2.putText(info_panel, "Channel Min / Max Stats", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y_offset += line_height + 5

    for name, (channel, color) in channels.items():
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(channel)

        # چاپ در ترمینال
        print(f"🔹 [{name}]")
        print(
            f"   ▫️ حداقل (Min): {int(min_val):3d}  در  X={min_loc[0]}, Y={min_loc[1]}")
        print(
            f"   ▫️ حداکثر (Max): {int(max_val):3d}  در  X={max_loc[0]}, Y={max_loc[1]}")

        # رسم نقاط روی تصویر (نقطه کوچکتر برای Min، بزرگتر برای Max)
        cv2.circle(marked_frame, min_loc, 2, color, -1)
        cv2.circle(marked_frame, max_loc, 4, color, -1)

        # متن مقادیر جهت نمایش در پنل
        text_min = f"{name} Min: {int(min_val):3d} @ ({min_loc[0]},{min_loc[1]})"
        text_max = f"{name} Max: {int(max_val):3d} @ ({max_loc[0]},{max_loc[1]})"

        if y_offset + (line_height * 2) < h:
            cv2.putText(info_panel, text_min, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            y_offset += line_height
            cv2.putText(info_panel, text_max, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            y_offset += line_height + 4

    print("=" * 65)

    # اتصال پنل اطلاعات به سمت راست تصویر اصلی
    combined_display = np.hstack((marked_frame, info_panel))

    win_name = "Marked Image with Stats Panel"
    cv2.imshow(win_name, combined_display)

    # ثبت کلیک موس
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < w:
                b_val, g_val, r_val = frame[y, x]
                h_val, s_val, v_val = h_chan[y, x], s_chan[y, x], v_chan[y, x]
                l_val, a_val, b_l_val = l_chan[y,
                                               x], a_chan[y, x], b_lab_chan[y, x]

                print(f"📍 کلیک در (X={x:4d}, Y={y:4d}) | "
                      f"RGB: (R={r_val:3d}, G={g_val:3d}, B={b_val:3d}) | "
                      f"HSV: (H={h_val:3d}, S={s_val:3d}, V={v_val:3d}) | "
                      f"LAB: (L={l_val:3d}, a={a_val:3d}, b={b_l_val:3d})")

    cv2.setMouseCallback(win_name, click_event)


def inspect_images_in_folder(input_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    if os.path.isdir(input_path):
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path)
                       if f.lower().endswith(valid_extensions)]
        image_paths.sort()
    elif os.path.isfile(input_path) and input_path.lower().endswith(valid_extensions):
        image_paths = [input_path]
    else:
        print("❌ مسیر معتبری برای فایل یا پوشه تصاویر یافت نشد!")
        return

    if not image_paths:
        print("❌ هیچ تصویری در پوشه موردنظر یافت نشد.")
        return

    total_images = len(image_paths)
    print(f"📂 تعداد {total_images} تصویر جهت پردازش یافت شد.\n")

    for idx, img_path in enumerate(image_paths, start=1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️ تصویر خوانده نشد: {img_path}")
            continue

        file_name = os.path.basename(img_path)
        process_single_frame(
            frame, title_prefix=f"{idx}/{total_images} - {file_name}")

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("\n⏹ پردازش متوقف شد.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    default_dir = "C:/Users/win/Desktop/SunTrackerSplit/SunTracker-Refactor/Refactor/Images"
    path = sys.argv[1] if len(sys.argv) > 1 else default_dir
    inspect_images_in_folder(path)
