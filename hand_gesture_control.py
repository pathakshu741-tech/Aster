"""
hand_gesture_control.py
------------------------
Drop this file next to your assistant's main .py file.

Uses MediaPipe's current "Tasks" API (mp.tasks.vision.HandLandmarker).

Recognized gestures (pass these strings as keys in your GESTURE_ACTIONS
dict in nova_app.py):

    fist         - all fingers curled, thumb tucked in
    open_palm    - all five fingers extended
    point        - index finger only
    peace        - index + middle ("victory" sign)
    three        - index + middle + ring
    four         - index + middle + ring + pinky (thumb tucked, fingers together)
    spock        - same four fingers as "four", but spread with a visible gap
                   between middle and ring (the Vulcan salute)
    thumbs_up    - fist with thumb extended upward
    thumbs_down  - fist with thumb extended downward
    pinky        - pinky finger only
    call_me      - thumb + pinky extended ("shaka" / hang-loose / phone sign)
    l_shape      - thumb + index extended, forming an L
    rock_on      - thumb + index + pinky extended
    thumb_middle - thumb + middle finger extended, others curled
    ok_sign      - thumb tip and index tip pinched together, other three
                   fingers extended

Any gesture not currently mapped in GESTURE_ACTIONS is simply ignored by
handle_gesture(), so you can add new ones here without breaking anything
that doesn't use them yet.
"""

import os
import math
import time
import threading
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Full set of gesture names this controller can emit. Exposed so the app
# (or a settings screen) can list them without duplicating this list.
ALL_GESTURES = (
    "fist", "open_palm", "point", "peace", "three", "four", "spock",
    "thumbs_up", "thumbs_down", "pinky", "call_me", "l_shape",
    "rock_on", "thumb_middle", "ok_sign",
)

# Normalized-coordinate (0..1 of frame size) tuning constants.
OK_SIGN_PINCH_THRESHOLD = 0.06   # how close thumb tip + index tip must be
THUMB_VERTICAL_MARGIN = 0.03     # how far tip must be above/below its knuckle
SPOCK_GAP_RATIO = 1.6            # how much wider the middle-ring gap must be
                                  # than the neighboring gaps to count as "spread"


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~7MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)


class HandGestureController:
    """
    Runs a webcam capture loop on a background thread, detects hand
    landmarks with MediaPipe, classifies a small fixed set of gestures,
    and calls on_gesture(name) whenever a new gesture is confidently held.

    If on_frame is given, it's called on every processed frame with a BGR
    numpy array (landmark dots + gesture label drawn on it when
    show_preview=True), so the caller can render it — e.g. into a Tkinter
    Label — instead of a separate cv2.imshow window.

    IMPORTANT: on_gesture and on_frame both run on a background thread.
    If they touch Tkinter widgets, wrap the call in window.after(0, ...).
    """

    def __init__(self, on_gesture, camera_index=0, cooldown=1.2,
                 show_preview=True, on_frame=None, show_cv_window=False):
        self.on_gesture = on_gesture
        self.on_frame = on_frame
        self.camera_index = camera_index
        self.cooldown = cooldown
        self.show_preview = show_preview       # draw landmark dots + label on the frame
        self.show_cv_window = show_cv_window   # also pop a separate cv2 debug window

        self._running = False
        self._thread = None
        self._last_gesture = None
        self._last_trigger_time = 0.0

    def start(self):
        if self._running:
            return
        _ensure_model()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self):
        return self._running

    @staticmethod
    def _fingers_up(landmarks, handedness_label):
        """Returns [thumb, index, middle, ring, pinky] as 1 (extended) / 0 (curled)."""
        fingers = []
        if handedness_label == "Right":
            fingers.append(1 if landmarks[4][0] < landmarks[3][0] else 0)
        else:
            fingers.append(1 if landmarks[4][0] > landmarks[3][0] else 0)
        for tip_id in (8, 12, 16, 20):
            fingers.append(1 if landmarks[tip_id][1] < landmarks[tip_id - 2][1] else 0)
        return fingers

    @staticmethod
    def _classify(landmarks, fingers):
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]   # thumb's base knuckle, used for up/down direction
        index_tip = landmarks[8]

        thumb_extended = fingers[0] == 1
        non_thumb_total = sum(fingers[1:])

        # OK sign: thumb and index tips pinched together while the other
        # three fingers are extended. Checked first since the raw
        # up/down finger pattern for this pose can otherwise look like
        # "point" or "three" depending on how tightly pinched it is.
        pinch_dist = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        if pinch_dist < OK_SIGN_PINCH_THRESHOLD and fingers[2] and fingers[3] and fingers[4]:
            return "ok_sign"

        # Thumb-only poses: distinguish fist / thumbs_up / thumbs_down by
        # where the thumb tip sits relative to its own knuckle (screen y
        # increases downward, so "above" means a smaller y value).
        if non_thumb_total == 0:
            if not thumb_extended:
                return "fist"
            if thumb_tip[1] < thumb_mcp[1] - THUMB_VERTICAL_MARGIN:
                return "thumbs_up"
            if thumb_tip[1] > thumb_mcp[1] + THUMB_VERTICAL_MARGIN:
                return "thumbs_down"
            return "thumbs_up"  # thumb extended but roughly level — default to up

        total = sum(fingers)
        if total == 5:
            return "open_palm"
        if fingers == [0, 1, 0, 0, 0]:
            return "point"
        if fingers == [0, 1, 1, 0, 0]:
            return "peace"
        if fingers == [0, 1, 1, 1, 0]:
            return "three"
        if fingers == [0, 1, 1, 1, 1]:
            # "four" and "spock" share the same raised-finger pattern —
            # spock is distinguished only by a deliberate gap between the
            # middle and ring fingertips, wider than the other two gaps.
            gap_index_middle = abs(landmarks[8][0] - landmarks[12][0])
            gap_middle_ring = abs(landmarks[12][0] - landmarks[16][0])
            gap_ring_pinky = abs(landmarks[16][0] - landmarks[20][0])
            if (gap_middle_ring > gap_index_middle * SPOCK_GAP_RATIO and
                    gap_middle_ring > gap_ring_pinky * SPOCK_GAP_RATIO):
                return "spock"
            return "four"
        if fingers == [0, 0, 0, 0, 1]:
            return "pinky"
        if fingers == [1, 0, 0, 0, 1]:
            return "call_me"
        if fingers == [1, 1, 0, 0, 0]:
            return "l_shape"
        if fingers == [1, 1, 0, 0, 1]:
            return "rock_on"
        if fingers == [1, 0, 1, 0, 0]:
            return "thumb_middle"
        return None

    def _loop(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("HandGestureController: could not open camera", self.camera_index)
            self._running = False
            return

        with HandLandmarker.create_from_options(options) as landmarker:
            start_time = time.time()
            while self._running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start_time) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                gesture = None
                if result.hand_landmarks:
                    pts = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
                    handedness_label = "Right"
                    if result.handedness and result.handedness[0]:
                        handedness_label = result.handedness[0][0].category_name
                    fingers = self._fingers_up(pts, handedness_label)
                    gesture = self._classify(pts, fingers)

                    if self.show_preview:
                        h, w, _ = frame.shape
                        for (x, y) in pts:
                            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 255), -1)

                now = time.time()
                if gesture and now - self._last_trigger_time > self.cooldown:
                    self._last_trigger_time = now
                    self._last_gesture = gesture
                    try:
                        self.on_gesture(gesture)
                    except Exception as e:
                        print("Gesture callback error:", e)
                elif gesture is None:
                    self._last_gesture = None

                if self.show_preview and gesture:
                    cv2.putText(frame, gesture, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if self.on_frame is not None:
                    try:
                        self.on_frame(frame)
                    except Exception as e:
                        print("Frame callback error:", e)

                if self.show_cv_window:
                    cv2.imshow("Hand Gesture Control (press ESC to close preview)", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        cap.release()
        if self.show_cv_window:
            cv2.destroyAllWindows()
        self._running = False