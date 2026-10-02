import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
# Save the model next to this script, no matter where it's run from
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
)
LIPS = vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS

UPPER_LIP, LOWER_LIP = 13, 14  # inner lips (center)
LEFT_CORNER, RIGHT_CORNER = 61, 291  # mouth corners

OPEN_THRESHOLD = 0.05  # opening / width ratio above this = mouth open (tune this!)
SPEAK_HOLD_SEC = 0.2  # stay "speaking" this long after the mouth last opened


def ensure_model():
    """Download the face landmarker model if it isn't already on disk."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return
    print(f"Downloading model to {MODEL_PATH} ...")
    tmp_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        os.replace(tmp_path, MODEL_PATH)  # only keep it if the download finished
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise SystemExit(f"Could not download model: {e}")
    print("Done.")


def dist(a, b, w, h):
    return (((a.x - b.x) * w) ** 2 + ((a.y - b.y) * h) ** 2) ** 0.5


def mouth_ratio(face, w, h):
    """How open the mouth is, divided by its width (so distance to camera doesn't matter)."""
    opening = dist(face[UPPER_LIP], face[LOWER_LIP], w, h)
    width = dist(face[LEFT_CORNER], face[RIGHT_CORNER], w, h)
    return opening / width if width else 0.0


def main():
    ensure_model()

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    start = time.monotonic()
    last_open = -999.0
    history = deque(maxlen=5)  # smooths out jitter

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int((now - start) * 1000),
            )

            if result.face_landmarks:
                face = result.face_landmarks[0]
                history.append(mouth_ratio(face, w, h))
                ratio = sum(history) / len(history)

                is_open = ratio > OPEN_THRESHOLD
                if is_open:
                    last_open = now
                speaking = now - last_open < SPEAK_HOLD_SEC

                if speaking:
                    cv2.putText(
                        frame,
                        "SPEAKING",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 200, 255),
                        2,
                    )
            else:
                history.clear()

            cv2.imshow("Mouth detector", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
