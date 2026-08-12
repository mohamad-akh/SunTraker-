import cv2
import numpy as np
import sys
import os

INSPECTOR_WIN_NAME = "Pixel Inspector"

def imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    """
    تابع جایگزین برای cv2.imread جهت پشتیبانی کامل از حروف فارسی و Unicode در مسیر فایل
    """
    try:
        # خواندن فایل به‌صورت آرایه بایت‌های خام
        img_array = np.fromfile(filepath, dtype=np.uint8)
        # دکود کردن بایت‌ها به تصویر OpenCV
        return cv2.imdecode(img_array, flags)
    except Exception as e:
        return None

def create_pixel_inspector_panel(r, g, b, h, s, v, l, a, b_lab, x, y):
    panel_w, panel_h = 400, 320
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    color_preview = np.full((60, panel_w - 20, 3), (b, g, r), dtype=np.uint8)
    panel[15:75, 10:panel_w-10] = color_preview
    
    cv2.rectangle(panel, (10, 15), (panel_w - 10, 75), (200, 200, 200), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, f"Position: X={x}, Y={y}", (15, 100), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"RGB:  R={r:3d}  G={g:3d}  B={b:3d}", (15, 140), font, 0.5, (100, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"HSV:  H={h:3d}  S={s:3d}  V={v:3d}", (15, 180), font, 0.5, (255, 200, 100), 1, cv2.LINE_AA)
    cv2.putText(panel, f"LAB:  L={l:3d}  a={a:3d}  b={b_lab:3d}", (15, 220), font, 0.5, (255, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "Move/Click mouse on main image", (15, 280), font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    
    return panel

def process_single_frame(frame, title_prefix=""):
    b_chan, g_chan, r_chan = cv2.split(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_lab_chan = cv2.split(lab)

    channels = {
        "Red (R)": (r_chan, (0, 0, 255)),
        "Green (G)": (g_chan, (0, 255, 0)),
        "Blue (B)": (b_chan, (255, 0, 0)),
        "Hue (H)": (h_chan, (255, 100, 0)),
        "Sat (S)": (s_chan, (0, 255, 128)),
        "Val (V)": (v_chan, (128, 0, 255)),
        "Light (L)": (l_chan, (255, 255, 0)),
        "a* (G-R)": (a_chan, (255, 0, 255)),
        "b* (B-Y)": (b_lab_chan, (0, 255, 255))
    }

    marked_frame = frame.copy()
    h, w = frame.shape[:2]

    panel_width = 380
    info_panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    print("=" * 65)
    print(f"📊 آمار حداقل و حداکثر کانال‌ها [{title_prefix}]:")
    print("=" * 65)

    y_offset = 25
    line_height = 20

    cv2.putText(info_panel, "Channel Min / Max Stats", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y_offset += line_height + 5

    for name, (channel, color) in channels.items():
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(channel)

        print(f"🔹 [{name}]")
        print(f"   ▫️ حداقل (Min): {int(min_val):3d}  در  X={min_loc[0]}, Y={min_loc[1]}")
        print(f"   ▫️ حداکثر (Max): {int(max_val):3d}  در  X={max_loc[0]}, Y={max_loc[1]}")

        cv2.circle(marked_frame, min_loc, 2, color, -1)
        cv2.circle(marked_frame, max_loc, 4, color, -1)

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

    combined_display = np.hstack((marked_frame, info_panel))
    win_name = "Marked Image with Stats Panel"
    
    cv2.imshow(win_name, combined_display)

    init_x, init_y = w // 2, h // 2
    b_val, g_val, r_val = frame[init_y, init_x]
    inspector_img = create_pixel_inspector_panel(
        r_val, g_val, b_val, 
        h_chan[init_y, init_x], s_chan[init_y, init_x], v_chan[init_y, init_x],
        l_chan[init_y, init_x], a_chan[init_y, init_x], b_lab_chan[init_y, init_x],
        init_x, init_y
    )
    cv2.imshow(INSPECTOR_WIN_NAME, inspector_img)

    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE:
            if x < w and y < h:
                b_v, g_v, r_v = frame[y, x]
                h_v, s_v, v_v = h_chan[y, x], s_chan[y, x], v_chan[y, x]
                l_v, a_v, b_l_v = l_chan[y, x], a_chan[y, x], b_lab_chan[y, x]

                updated_inspector = create_pixel_inspector_panel(
                    r_v, g_v, b_v, h_v, s_v, v_v, l_v, a_v, b_l_v, x, y
                )
                cv2.imshow(INSPECTOR_WIN_NAME, updated_inspector)

                if event == cv2.EVENT_LBUTTONDOWN:
                    print(f"📍 کلیک در (X={x:4d}, Y={y:4d}) | "
                          f"RGB: ({r_v:3d}, {g_v:3d}, {b_v:3d}) | "
                          f"HSV: ({h_v:3d}, {s_v:3d}, {v_v:3d}) | "
                          f"LAB: ({l_v:3d}, {a_v:3d}, {b_l_v:3d})")

    cv2.setMouseCallback(win_name, mouse_event)

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

    cv2.namedWindow(INSPECTOR_WIN_NAME, cv2.WINDOW_AUTOSIZE)

    total_images = len(image_paths)
    print(f"📂 تعداد {total_images} تصویر جهت پردازش یافت شد.\n")

    for idx, img_path in enumerate(image_paths, start=1):
        # تغییر اصلی: استفاده از تابع اختصاصی imread_unicode به جای cv2.imread
        frame = imread_unicode(img_path)
        if frame is None:
            print(f"⚠️ تصویر خوانده نشد (احتمالا فایل خراب یا نامعتبر است): {img_path}")
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