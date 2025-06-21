import cv2
import mediapipe as mp
import pyautogui
import time
import math

pyautogui.FAILSAFE = False

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_time = 0
paused = False

def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img_h, img_w, _ = img.shape

    key = cv2.waitKey(1)
    if key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")
        time.sleep(0.3)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get all key landmarks
            def landmark_pos(landmark_enum):
                lm = hand_landmarks.landmark[landmark_enum]
                return int(lm.x * img_w), int(lm.y * img_h)

            index_tip = landmark_pos(mp_hands.HandLandmark.INDEX_FINGER_TIP)
            middle_tip = landmark_pos(mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
            thumb_tip = landmark_pos(mp_hands.HandLandmark.THUMB_TIP)

            screen_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_w)
            screen_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_h)

            if not paused:
                # Move cursor
                pyautogui.moveTo(screen_x, screen_y)

                # Left click: Thumb + Index
                if get_distance(index_tip, thumb_tip) < 30:
                    pyautogui.click()
                    time.sleep(0.3)

                # Right click: Index + Middle
                elif get_distance(index_tip, middle_tip) < 30:
                    pyautogui.rightClick()
                    time.sleep(0.3)

                # Scroll: Two fingers up (Index and Middle)
                elif index_tip[1] < img_h // 2 and middle_tip[1] < img_h // 2:
                    pyautogui.scroll(20)
                    time.sleep(0.1)

            # Draw tracking circle
            cv2.circle(img, index_tip, 10, (0, 255, 0), cv2.FILLED)

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, f'Press P to {"Resume" if paused else "Pause"}', (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AI Virtual Mouse", img)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
