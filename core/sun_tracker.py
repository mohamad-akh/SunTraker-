import cv2
import numpy as np
from datetime import datetime
import time
import os

# ==========================================================
# Smart Sun Tracker V3
# ==========================================================

# -------------------------------
# Detection Parameters
# -------------------------------

MIN_SUN_AREA_RATIO = 0.0002
MAX_SUN_AREA_RATIO = 0.10

MIN_CIRCULARITY = 0.65
MIN_SOLIDITY = 0.90

MIN_ASPECT_RATIO = 0.60
MAX_ASPECT_RATIO = 1.40

MIN_VALUE_AFTER_OTSU = 180

LOCAL_CONTRAST_LIMIT = 20

# -------------------------------
# Tracking
# -------------------------------

TRACKING_WINDOW = 200
TRACK_LOST_FRAMES = 8

START_CONFIRM_FRAMES = 3
STOP_CONFIRM_FRAMES = 5

COOLDOWN_SECONDS = 3.0

# -------------------------------
# Recording
# -------------------------------

RECORDING_DIR = "sun_recordings"

os.makedirs(RECORDING_DIR, exist_ok=True)


# ==========================================================
# ROI Helper
# ==========================================================

def crop_roi(frame, center, window):

    h, w = frame.shape[:2]

    cx, cy = center

    half = window // 2

    x1 = max(0, cx-half)
    y1 = max(0, cy-half)

    x2 = min(w, cx+half)
    y2 = min(h, cy+half)

    roi = frame[y1:y2, x1:x2]

    return roi, x1, y1


# ==========================================================
# Sun Detection
# ==========================================================

def detect_sun_robust(frame, offset_x=0, offset_y=0):

    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]

    if offset_x == 0 and offset_y == 0:
        sky_limit = int(h * 0.90)
        frame = frame[:sky_limit]
        h, w = frame.shape[:2]

    total_pixels = h * w

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    value = hsv[:, :, 2]

    blur = cv2.GaussianBlur(value, (9,9), 0)

    # ----------------------------
    # OTSU
    # ----------------------------

    _, otsu = cv2.threshold(

        blur,

        0,

        255,

        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    # ----------------------------
    # Adaptive
    # ----------------------------

    adaptive = cv2.adaptiveThreshold(

        blur,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        61,

        -4
    )

    # ----------------------------
    # Bright Pixels
    # ----------------------------

    bright = cv2.inRange(

        value,

        MIN_VALUE_AFTER_OTSU,

        255
    )

    # ----------------------------
    # Final Mask
    # ----------------------------

    mask = cv2.bitwise_or(

        otsu,

        adaptive
    )

    mask = cv2.bitwise_and(

        mask,

        bright
    )

    kernel1 = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (5,5)
    )

    kernel2 = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (13,13)
    )

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel1
    )

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel2
    )

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None

    best_score = -1

    # -------------------------------------------------
    # Candidate Selection
    # -------------------------------------------------

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < total_pixels * MIN_SUN_AREA_RATIO:
            continue

        if area > total_pixels * MAX_SUN_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter <= 0:
            continue

        circularity = (
            4.0 * np.pi * area
    ) / (perimeter * perimeter)

        if circularity < MIN_CIRCULARITY:
            continue

        hull = cv2.convexHull(cnt)

        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            continue

        solidity = area / hull_area

        if solidity < MIN_SOLIDITY:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bh == 0:
           continue

        aspect_ratio = bw / float(bh)

        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        object_mask = np.zeros_like(mask)

        cv2.drawContours(
            object_mask,
            [cnt],
            -1,
            255,
            -1
    )

        mean_value = cv2.mean(
            value,
            mask=object_mask
    )[0]

        # ---------------------------------
        # Local Contrast
        # ---------------------------------

        ring_mask = cv2.dilate(

            object_mask,

            cv2.getStructuringElement(

                cv2.MORPH_ELLIPSE,

                (31, 31)
    )
    )

        ring_mask = cv2.subtract(
            ring_mask,
            object_mask
    )

        background_value = cv2.mean(
            value,
            mask=ring_mask
    )[0]

        local_contrast = (
            mean_value -
            background_value
    )

        if local_contrast < LOCAL_CONTRAST_LIMIT:
            continue

        mean_bgr = cv2.mean(
            frame,
            mask=object_mask
    )[:3]

        (center_tmp, radius) = cv2.minEnclosingCircle(cnt)

        radius = int(radius)

        score = (

            mean_value * 1.8 +

            local_contrast * 8.0 +

            circularity * 250 +

            solidity * 250 +

            np.sqrt(area)
    )

        if score > best_score:

            best_score = score

            best = {

                "center": (
                    cx + offset_x,
                    cy + offset_y
    ),

                "radius_est": radius,

                "pixel_count": int(area),

                "mean_bgr": mean_bgr,

                "mean_value": mean_value,

                "local_contrast": local_contrast,

                "circularity": circularity,

                "solidity": solidity,

                "score": score

            }

    return best


# ==========================================================
# Smart Sun Tracker
# ==========================================================

class SunTracker:

    def __init__(self, video_source=0):
        if isinstance(video_source, str):
            self.cap = cv2.VideoCapture(video_source)
        else:
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('Y','U','Y','V'))
        #self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            raise RuntimeError(" دوربین باز نشد!")

        self.frame_width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

        self.frame_height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

        self.fps = max(
            1,
            int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
    )

        self.recording = False
        self.out = None

        self.last_sun_time = 0

        # Multi-frame confirmation

        self.detect_counter = 0
        self.lost_counter = 0

        # ROI Tracking

        self.last_center = None
        self.tracking = False
        self.track_lost = 0

        # ROI Tracking Parameters

        self.search_window = TRACKING_WINDOW

    # ======================================================

    def start_recording(self):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = os.path.join(

            RECORDING_DIR,

            f"sun_{timestamp}.avi"
    )

        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        self.out = cv2.VideoWriter(

            filename,

            fourcc,

            self.fps,

            (

                self.frame_width,

                self.frame_height
    )
    )

        if not self.out.isOpened():

            raise RuntimeError(

                " ایجاد فایل ویدئو ناموفق بود."
    )

        self.recording = True

        print(f" Recording : {filename}")

    # ======================================================

    def stop_recording(self):

        if self.out is not None:

            self.out.release()

            self.out = None

        self.recording = False

        print(" Recording Stopped")

    # ======================================================

    def search_sun(self, frame):

        # -----------------------------------------
        # ROI Tracking
        # -----------------------------------------

        if self.tracking and self.last_center is not None:

            roi, ox, oy = crop_roi(

                frame,

                self.last_center,

                self.search_window
    )

            result = detect_sun_robust(

                roi,

                ox,

                oy
    )

            if result is not None:

                self.track_lost = 0

                self.last_center = result["center"]

                return result

            self.track_lost += 1

            if self.track_lost >= TRACK_LOST_FRAMES:

                self.tracking = False

                self.last_center = None

        # -----------------------------------------
        # Full Search
        # -----------------------------------------

        result = detect_sun_robust(frame)

        if result is not None:

            self.tracking = True

            self.track_lost = 0

            self.last_center = result["center"]

        return result

    # ======================================================

    def run(self):

        print("===================================")
        print(" Smart Sun Tracker V3 ")
        print("===================================")

        try:

            while True:

                ret, frame = self.cap.read()

                if not ret or frame is None:

                    time.sleep(0.02)

                    continue

                current_time = time.time()

                sun = self.search_sun(frame)

                if sun is not None:

                    self.detect_counter += 1

                    self.lost_counter = 0

                else:

                    self.detect_counter = 0

                    self.lost_counter += 1

                detected = (

                    self.detect_counter >=

                    START_CONFIRM_FRAMES
    )

                # ---------------------------------------
                # Recording Logic
                # ---------------------------------------

                if detected:

                    self.last_sun_time = current_time

                    if not self.recording:

                        self.start_recording()

                else:

                    if (

                        self.recording

                        and

                        self.lost_counter >= STOP_CONFIRM_FRAMES

                        and

                        (current_time-self.last_sun_time)
                        > COOLDOWN_SECONDS
    ):

                        self.stop_recording()

                # ---------------------------------------
                # Draw Result
                # ---------------------------------------

                if detected:

                    cx, cy = sun["center"]

                    radius = sun["radius_est"]

                    cv2.circle(

                        frame,

                        (cx, cy),

                        radius,

                        (0,255,255),

                        2
    )

                    cv2.circle(

                        frame,

                        (cx, cy),

                        3,

                        (0,0,255),

                        -1
    )

                    cv2.putText(

                        frame,

                        f"Sun ({cx},{cy})",

                        (10,30),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (0,255,255),

                        2
    )

                    cv2.putText(

                        frame,

                        f"Score : {sun['score']:.1f}",

                        (10,60),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.60,

                        (255,255,0),

                        2
    )

                    cv2.putText(

                        frame,

                        f"Contrast : {sun['local_contrast']:.1f}",

                        (10,90),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.60,

                        (255,255,255),

                        2
    )

                if self.recording:

                    status = "Recording"

                    color = (0,255,0)

                    self.out.write(frame)

                else:

                    status = "Waiting"

                    color = (0,0,255)

                cv2.putText(

                    frame,

                    status,

                    (10,self.frame_height-20),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.70,

                    color,

                    2
    )

                cv2.imshow(

                    "Smart Sun Tracker V4",

                    frame
    )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    break

        finally:

            if self.recording:
                self.stop_recording()

            self.cap.release()

            cv2.destroyAllWindows()

