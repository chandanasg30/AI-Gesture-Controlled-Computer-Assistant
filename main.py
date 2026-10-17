
import cv2
import mediapipe as mp
import pyautogui
import math
import time

pyautogui.FAILSAFE = True

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Gesture stabilization
candidate_gesture = "None"
stable_gesture = "None"
stable_frames = 0
last_action_gesture = "None"
last_action_time = 0
COOLDOWN = 0.8
REQUIRED_FRAMES = 5

controls_paused = False

# Smooth cursor movement
cursor_x, cursor_y = screen_w // 2, screen_h // 2
SMOOTHING = 0.35


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def detect_fingers(lm):
    # A finger is up when its tip is above its PIP joint.
    # The small margin reduces noisy classifications.
    return [
        lm[8].y < lm[6].y - 0.02,
        lm[12].y < lm[10].y - 0.02,
        lm[16].y < lm[14].y - 0.02,
        lm[20].y < lm[18].y - 0.02
    ]


def classify_gesture(lm):
    fingers = detect_fingers(lm)
    count = sum(fingers)

    # Estimate whether the thumb is extended
    thumb_extended = (
        distance(lm[4], lm[5])
        > distance(lm[3], lm[5]) * 1.15
    )

    # Pinch takes priority over cursor movement
    if distance(lm[4], lm[8]) < 0.045:
        return "Pinch"

    # Four fingers extended with thumb extended
    if count == 4 and thumb_extended:
        return "Open Palm"

    # Four fingers extended, thumb folded
    if count == 4 and not thumb_extended:
        return "Scroll Up"

    # Index finger only
    if count == 1 and fingers[0]:
        return "Cursor"

    # Two fingers: next slide
    if count == 2 and fingers[0] and fingers[1]:
        return "Next Slide"

    # Three fingers: previous slide
    if count == 3 and all(fingers[:3]):
        return "Previous Slide"

    # Thumb gestures: other four fingers folded
    if count == 0 and thumb_extended:
        if lm[4].y < lm[2].y - 0.04:
            return "Volume Up"
        if lm[4].y > lm[2].y + 0.04:
            return "Volume Down"
        return "Unknown"

    # Closed fist
    if count == 0 and not thumb_extended:
        return "Play/Pause"

    return "Unknown"


def perform_action(gesture):
    global controls_paused

    if gesture == "Open Palm":
        controls_paused = not controls_paused

    elif controls_paused:
        return "Controls Paused"

    elif gesture == "Pinch":
        pyautogui.click()

    elif gesture == "Next Slide":
        pyautogui.press("right")

    elif gesture == "Previous Slide":
        pyautogui.press("left")

    elif gesture == "Play/Pause":
        pyautogui.press("playpause")

    elif gesture == "Volume Up":
        pyautogui.press("volumeup")

    elif gesture == "Volume Down":
        pyautogui.press("volumedown")

    elif gesture == "Scroll Up":
        pyautogui.scroll(3)

    return gesture


if not cap.isOpened():
    raise RuntimeError("Webcam unavailable. Check camera permissions.")

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            detected = "None"
            status = "Show your hand"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark

                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                detected = classify_gesture(lm)

                # Stabilize the gesture across multiple frames
                if detected == candidate_gesture:
                    stable_frames += 1
                else:
                    candidate_gesture = detected
                    stable_frames = 1

                if stable_frames >= REQUIRED_FRAMES:
                    stable_gesture = candidate_gesture

                now = time.time()

                # Cursor moves continuously
                if stable_gesture == "Cursor" and not controls_paused:
                    target_x = int(lm[8].x * screen_w)
                    target_y = int(lm[8].y * screen_h)

                    cursor_x = int(
                        cursor_x + (target_x - cursor_x) * SMOOTHING
                    )
                    cursor_y = int(
                        cursor_y + (target_y - cursor_y) * SMOOTHING
                    )

                    pyautogui.moveTo(cursor_x, cursor_y)
                    status = "Cursor Mode"

                elif stable_gesture == "None":
                    last_action_gesture = "None"
                    status = "Show a gesture"

                elif stable_gesture == "Unknown":
                    status = "Gesture not recognized"

                elif stable_gesture == "Open Palm":
                    if (
                        last_action_gesture != "Open Palm"
                        and now - last_action_time >= COOLDOWN
                    ):
                        perform_action("Open Palm")
                        last_action_gesture = "Open Palm"
                        last_action_time = now

                    status = (
                        "Controls Paused"
                        if controls_paused
                        else "Controls Active"
                    )

                elif stable_gesture != "Cursor":
                    if (
                        stable_gesture != last_action_gesture
                        and now - last_action_time >= COOLDOWN
                    ):
                        status = perform_action(stable_gesture)
                        last_action_gesture = stable_gesture
                        last_action_time = now
                    else:
                        status = stable_gesture

                # Show both detected and stabilized gesture
                cv2.putText(
                    frame,
                    f"Detected: {detected}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Stable: {stable_gesture}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 200, 0),
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

            else:
                candidate_gesture = "None"
                stable_gesture = "None"
                stable_frames = 0
                last_action_gesture = "None"
                status = "No hand detected"

            cv2.imshow("AI Gesture Assistant", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
