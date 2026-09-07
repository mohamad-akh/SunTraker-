import os
import cv2
import numpy as np
import time

# ==========================================
# 0. Configurable Parameters (Algorithm)
# ==========================================
GAUSSIAN_KERNEL = (9, 9)
DELTA_THRESHOLD = 4
ROI_WIDTH_RATIO = 0.5
ROI_HEIGHT_RATIO = 0.4

# Hough Circle Parameters
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 10
HOUGH_MIN_RADIUS = 5
HOUGH_MAX_RADIUS = 100

# Parameters copied from redesigned.ipynb sky analysis
NIGHT_BRIGHTNESS_THRESH = 58
BLUE_THRESH = 150

# ==========================================
# 0b. Configurable Parameters (UI/Display)
# ==========================================
WINDOW_NAME = "Sun Detection Pipeline - Modular View"
MONITOR_WIDTH = 1920
MONITOR_HEIGHT = 1080


# ==========================================
# 1. Sky & Cloud Analysis
#    Logic copied from redesigned.ipynb
# ==========================================
def analyze_sky_and_clouds(frame):
    img = frame
    if img is None:
        return False, {"error": "Error loading image"}

    # کانال L از LAB برای سنجش روشنایی شب/روز
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]

    # بررسی شب بودن بر اساس ردیف ۱۰ درصد ارتفاع تصویر
    h, w = img.shape[:2]
    mean_dark = np.mean(l_channel[int(0.1 * h)])

    if mean_dark < NIGHT_BRIGHTNESS_THRESH:
        return False, {
            "cloud_status": "Night / N/A",
            "brightness": mean_dark,
        }

    # پیدا کردن موقعیت روشن‌ترین نقطه در روز
    _, _, _, sun_center = cv2.minMaxLoc(
        cv2.GaussianBlur(l_channel, (11, 11), 0)
    )

    # ماسک دایره‌ای اطراف نقطه روشن
    mask = cv2.circle(
        np.zeros_like(l_channel), sun_center, 50, 255, -1
    ) > 0

    # کانال آبی در BGR
    blue_channel = img[:, :, 0]
    mean_blue = float(np.mean(blue_channel[mask]))

    # تشخیص ابر تاریک / حالت عادی
    if mean_blue < BLUE_THRESH:
        cloud_status = "Dark Clouds"
    else:
        cloud_status = "Clear / Moderate Sky"

    return True, {
        "cloud_status": cloud_status,
        "brightness": mean_blue,
    }


def find_brightest_point(l_channel):
    blurred_l = cv2.GaussianBlur(l_channel, GAUSSIAN_KERNEL, 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred_l)
    return max_loc, blurred_l


def create_roi(image_shape, max_point):
    h, w = image_shape[:2]
    x_max, y_max = max_point
    roi_w = int(w * ROI_WIDTH_RATIO)
    roi_h = int(h * ROI_HEIGHT_RATIO)
    x1 = max(0, x_max - roi_w // 2)
    y1 = max(0, y_max - roi_h // 2)
    x2 = min(w, x_max + roi_w // 2)
    y2 = min(h, y_max + roi_h // 2)
    return x1, y1, x2, y2


def heavy_threshold(l_roi):
    max_val = np.max(l_roi)
    thresh_val = max(0, max_val - DELTA_THRESHOLD)
    _, thresh_mask = cv2.threshold(
        l_roi, thresh_val, 255, cv2.THRESH_BINARY
    )
    return thresh_mask


# ==========================================
# 2. Hough Detection
#    The exact same current Hough logic is
#    duplicated for the two future modes.
# ==========================================
def detect_hough_white(thresh_roi):
    circles = cv2.HoughCircles(
        thresh_roi,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )
    return circles


def detect_hough_dark(thresh_roi):
    circles = cv2.HoughCircles(
        thresh_roi,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )
    return circles


def select_best_circle(circles, thresh_roi, l_roi, detection_mode):
    if circles is None:
        return None

    circles = np.uint16(np.around(circles))
    best_circle, best_score = None, -1

    for circle in circles[0, :]:
        cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
        mask = np.zeros_like(thresh_roi, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)

        bright_pixels = cv2.countNonZero(
            cv2.bitwise_and(thresh_roi, thresh_roi, mask=mask)
        )

        if detection_mode == "Dark Clouds Hough":
            if bright_pixels == 0:
                continue

            circle_area = np.pi * (r ** 2)
            density = bright_pixels / circle_area
            mean_l_inside = cv2.mean(l_roi, mask=mask)[0]

            score = (
                bright_pixels * density * 0.5
            ) + (
                mean_l_inside * 1.2
            )

            if score > 150.0 and score > best_score:
                best_score = score
                best_circle = (
                    cx, cy, r, score
                )
        else:
            pixel_count = bright_pixels

            # Current scoring logic kept unchanged for White Clouds.
            score = pixel_count * cv2.mean(l_roi, mask=mask)[0]

            if score > best_score:
                best_score = score
                best_circle = (cx, cy, r, score)

    return best_circle


# ==========================================
# 3. UI Helper Functions
# ==========================================
def resize_maintain_aspect(img, target_w, target_h, background_color=(0, 0, 0)):
    h, w = img.shape[:2]
    img_aspect = w / h
    target_aspect = target_w / target_h

    if img_aspect > target_aspect:
        new_w = target_w
        new_h = int(target_w / img_aspect)
    else:
        new_h = target_h
        new_w = int(target_h * img_aspect)

    resized_img = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_AREA
    )

    if len(resized_img.shape) == 2:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    final_img = np.full(
        (target_h, target_w, 3), background_color, dtype=np.uint8
    )
    off_y = (target_h - new_h) // 2
    off_x = (target_w - new_w) // 2
    final_img[off_y:off_y+new_h, off_x:off_x+new_w] = resized_img

    return final_img


def add_clean_label(img, text, position=(15, 30), font_scale=0.7, color=(0, 255, 255)):
    res = img.copy()
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    cv2.rectangle(
        res,
        (position[0]-5, position[1]-text_h-5),
        (position[0]+text_w+5, position[1]+5),
        (0, 0, 0),
        -1
    )
    cv2.putText(
        res,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA
    )
    return res


def visualize_shakil(
    img_bgr,
    l_chan,
    blurred_l,
    roi_bgr,
    thresh_roi,
    best_circle,
    roi_coords,
    max_pt,
    fps,
    detection_mode
):
    x1, y1, x2, y2 = roi_coords
    annotated_img = img_bgr.copy()

    cv2.rectangle(
        annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 3
    )
    cv2.circle(annotated_img, max_pt, 8, (0, 0, 255), -1)

    cv2.putText(
        annotated_img,
        f"Mode: {detection_mode}",
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    roi_annotated = roi_bgr.copy()

    if best_circle is not None:
        cx_roi, cy_roi, r, _ = best_circle
        cv2.circle(
            roi_annotated, (cx_roi, cy_roi), r, (0, 255, 0), 3
        )
        x_global, y_global = cx_roi + x1, cy_roi + y1
        cv2.circle(
            annotated_img, (x_global, y_global), r, (0, 255, 0), 3
        )
        cv2.circle(
            annotated_img, (x_global, y_global), 4, (0, 0, 255), -1
        )

        txt = f"Sun: ({x_global}, {y_global}), R:{r}"
        cv2.putText(
            annotated_img,
            txt,
            (x1, max(30, y1 - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
    else:
        cv2.putText(
            annotated_img,
            "No Sun Circle Detected",
            (x1, max(30, y1 - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    Num_Cols = 3
    Num_Rows = 2
    cell_w = int((MONITOR_WIDTH * 0.9) / Num_Cols)
    cell_h = int((MONITOR_HEIGHT * 0.8) / Num_Rows)

    cells = []
    cells.append(add_clean_label(
        resize_maintain_aspect(img_bgr, cell_w, cell_h),
        "1. Original BGR"
    ))
    cells.append(add_clean_label(
        resize_maintain_aspect(l_chan, cell_w, cell_h),
        "2. LAB - L Channel",
        color=(255, 200, 0)
    ))
    cells.append(add_clean_label(
        resize_maintain_aspect(roi_bgr, cell_w, cell_h),
        "3. ROI (Zoomed Original)"
    ))
    cells.append(add_clean_label(
        resize_maintain_aspect(thresh_roi, cell_w, cell_h),
        "4. Heavy Thresh Mask",
        color=(200, 200, 200)
    ))
    cells.append(add_clean_label(
        resize_maintain_aspect(roi_annotated, cell_w, cell_h),
        "5. ROI with Hough Detect"
    ))

    final_cell = resize_maintain_aspect(
        annotated_img, cell_w, cell_h
    )
    (fps_w, fps_h), _ = cv2.getTextSize(
        f"FPS: {fps:.1f}",
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        3
    )
    cv2.putText(
        final_cell,
        f"FPS: {fps:.1f}",
        (cell_w - fps_w - 20, cell_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
        cv2.LINE_AA
    )
    cells.append(add_clean_label(
        final_cell,
        "6. Final Full View",
        color=(0, 255, 0)
    ))

    row1 = np.hstack(cells[0:3])
    row2 = np.hstack(cells[3:6])
    grid = np.vstack([row1, row2])

    help_bar = np.zeros(
        (50, grid.shape[1], 3), dtype=np.uint8
    )
    cv2.putText(
        help_bar,
        "Press [ANY KEY] for Next Image | Press [ESC] to Exit",
        (int(grid.shape[1]/2) - 300, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    final_display = np.vstack([grid, help_bar])
    cv2.imshow(WINDOW_NAME, final_display)

    key = cv2.waitKey(0)
    if key == 27:
        return False
    return True


# ==========================================
# 4. Main Processing
# ==========================================
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping invalid: {image_path}")
        return True

    start_time = time.perf_counter()

    # ------------------------------------------
    # First: analyze sky/day/cloud condition
    # ------------------------------------------
    is_day, sky_details = analyze_sky_and_clouds(img)
    cloud_status = sky_details.get("cloud_status", "Unknown")
    brightness = sky_details.get("brightness", 0.0)

    # ------------------------------------------
    # Night: skip sun detection
    # ------------------------------------------
    if not is_day:
        elapsed_time = time.perf_counter() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

        print(
            f"{os.path.basename(image_path)} | "
            f"Mode: Night | Brightness: {brightness:.1f}"
        )

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        max_point, blurred_l = find_brightest_point(l_channel)
        x1, y1, x2, y2 = create_roi(img.shape, max_point)
        roi_bgr = img[y1:y2, x1:x2]
        l_roi = blurred_l[y1:y2, x1:x2]
        thresh_roi = heavy_threshold(l_channel[y1:y2, x1:x2])

        return visualize_shakil(
            img,
            l_channel,
            blurred_l,
            roi_bgr,
            thresh_roi,
            None,
            (x1, y1, x2, y2),
            max_point,
            fps,
            "Night (Skipped)"
        )

    # ------------------------------------------
    # Day: use current result.py Hough pipeline
    # ------------------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    max_point, blurred_l = find_brightest_point(l_channel)

    x1, y1, x2, y2 = create_roi(img.shape, max_point)
    l_roi = blurred_l[y1:y2, x1:x2]
    roi_bgr = img[y1:y2, x1:x2]
    thresh_roi = heavy_threshold(l_channel[y1:y2, x1:x2])

    if cloud_status == "Dark Clouds":
        detection_mode = "Dark Clouds Hough"
        circles = detect_hough_dark(thresh_roi)
    else:
        detection_mode = "White Clouds Hough"
        circles = detect_hough_white(thresh_roi)

    best_circle = select_best_circle(
        circles, thresh_roi, l_roi, detection_mode
    )

    elapsed_time = time.perf_counter() - start_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

    print(
        f"{os.path.basename(image_path)} | "
        f"Mode: {detection_mode} | "
        f"Sky: {cloud_status} | "
        f"Brightness: {brightness:.1f} | "
        f"Time: {elapsed_time*1000:.2f} ms | FPS: {fps:.2f}"
    )

    return visualize_shakil(
        img,
        l_channel,
        blurred_l,
        roi_bgr,
        thresh_roi,
        best_circle,
        (x1, y1, x2, y2),
        max_point,
        fps,
        detection_mode
    )


def process_folder(folder_path):
    if not os.path.exists(folder_path):
        print("Folder not found.")
        return

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    )
    cv2.resizeWindow(
        WINDOW_NAME,
        int(MONITOR_WIDTH * 0.9),
        int(MONITOR_HEIGHT * 0.9)
    )

    valid_exts = (
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff'
    )

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_exts):
            img_path = os.path.join(folder_path, filename)
            if not process_image(img_path):
                break

    cv2.destroyAllWindows()


# ==========================================
# Example Usage
# ==========================================
process_folder("C:/Users/win/Desktop/Output_Image")
