import cv2
import time
from mediapipe import solutions
from just_playback import Playback
from numpy import array, float64, linalg


face_mesh = solutions.face_mesh.FaceMesh()
left_eye = [33, 160, 158, 133, 153, 144]
right_eye = [362, 385, 387, 263, 373, 380]


def _ear(landmarks: list, eye: list, width: int, height: int) -> float64:
    pts = [
        (
            int(landmarks[i].x * width),
            int(landmarks[i].y * height),
        ) for i in eye
    ]
    A = linalg.norm(array(pts[1]) - array(pts[5]))
    B = linalg.norm(array(pts[2]) - array(pts[4]))
    C = linalg.norm(array(pts[0]) - array(pts[3]))
    return (A + B) / (2.0 * C)


alarm = False
cap = cv2.VideoCapture(0)
closed_at = None
player = Playback("alarm.mp3")
player.loop_at_end(True)
threshold = 0.22
sleep_time = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture camera")
        break

    _y, _x, _ = frame.shape
    res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        closed_at = None
        cv2.putText(frame, "Where are you?", (20, 40), 0, 1, (0,0,255), 2)
    else:
        lm = res.multi_face_landmarks[0].landmark

        ear_left = _ear(lm, left_eye, _x, _y)
        ear_right = _ear(lm, right_eye, _x, _y)

        val = (ear_left + ear_right) / 2.0

        if val < threshold:
            if not closed_at:
                closed_at = time.time()
            elif time.time() - closed_at > sleep_time:
                cv2.putText(frame, "Wake up!", (20, 80), 0, 1.2, (0,0,255), 3)
                if not alarm:
                    player.play()
                    alarm = True
        else:
            player.stop()
            alarm = False
            closed_at = None
            cv2.putText(frame, "Do something!", (20, 80), 0, 1.2, (0,255,0), 3)

    cv2.imshow("Sleep Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
player.stop()
