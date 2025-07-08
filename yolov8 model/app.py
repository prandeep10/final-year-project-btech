import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_PATH = "weights/best.pt"  # Your YOLOv8 model path
CAMERA_STATUS_URL = "http://192.168.43.125:3000/cameras/Camera1"
DARKNESS_THRESHOLD = 30
CONSECUTIVE_DARK_FRAMES_REQUIRED = 5
MIN_SERVER_UPDATE_INTERVAL = 2  # seconds
CAMERA_INDEX = 0  # Change if your camera is on another index

DESIRED_WIDTH = 640
DESIRED_HEIGHT = 480

RHINO_CONFIDENCE_THRESHOLD = 0.90  # 90%
RHINO_BOX_COLOR = (255, 0, 0)  # Blue in BGR

# --- Camera Thread Class ---
class CameraFeed:
    def __init__(self, src=CAMERA_INDEX, width=DESIRED_WIDTH, height=DESIRED_HEIGHT):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera with index {src}. Please check if it's connected.")
            raise IOError(f"Cannot open camera with index {src}. Please check if it's connected.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.read_lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())

    def start(self):
        self.thread.daemon = True
        self.thread.start()
        logger.info("Camera capture thread started.")
        return self

    def update(self):
        while True:
            if self.stopped:
                break
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            if not grabbed:
                logger.warning("Camera disconnected or failed to read frame. Stopping capture thread.")
                self.stopped = True
                break

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
        logger.info("Camera capture thread stopped.")

# --- Functions ---
def is_dark(frame, threshold=DARKNESS_THRESHOLD):
    if frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold

def update_status_on_server(rhino_detected, active):
    payload = {
        "status": {
            "active": active,
            "rhino_detection": rhino_detected
        }
    }
    try:
        response = requests.put(CAMERA_STATUS_URL, json=payload, timeout=2)
        response.raise_for_status()
        logger.info(f"PUT request successful: {payload}. Server responded with status code {response.status_code}.")
    except requests.exceptions.Timeout:
        logger.error("Failed to update status: Request timed out. Check server availability.")
    except requests.exceptions.ConnectionError:
        logger.error("Failed to update status: Connection error. Is the server running and accessible?")
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to update status: HTTP Error {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error during status update: {e}")

# --- Main Program ---
def main():
    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"YOLO model loaded from {MODEL_PATH}")
        logger.info(f"Model classes: {model.names}")

        half_precision = (model.device.type != 'cpu')
        if model.device.type == 'cpu':
            logger.warning("Model running on CPU. For best performance use GPU if available.")
        else:
            logger.info(f"Model running on {model.device.type}. Using half precision: {half_precision}")
    except Exception as e:
        logger.critical(f"Failed to load YOLO model: {e}")
        return

    # Initialize camera
    try:
        camera_feed = CameraFeed(src=CAMERA_INDEX, width=DESIRED_WIDTH, height=DESIRED_HEIGHT).start()
        time.sleep(1)  # Warm up
        if not camera_feed.grabbed:
            logger.error("No frames grabbed from camera after warm-up.")
            camera_feed.stop()
            return
    except IOError as e:
        logger.critical(f"Camera initialization failed: {e}")
        return

    consecutive_dark_frames = 0
    last_rhino_detected = False
    last_camera_active = True
    last_server_update_time = time.time()

    logger.info("Starting detection loop. Press 'q' to quit.")

    frame_counter = 0
    start_time = time.time()

    try:
        while True:
            grabbed, frame = camera_feed.read()
            if not grabbed or frame is None:
                if camera_feed.stopped:
                    logger.error("Camera stopped unexpectedly.")
                    break
                time.sleep(0.01)
                continue

            frame_counter += 1
            current_time = time.time()

            # Calculate FPS every second
            if current_time - start_time >= 1.0:
                fps = frame_counter / (current_time - start_time)
                logger.info(f"FPS: {fps:.2f}")
                frame_counter = 0
                start_time = current_time

            # Check darkness
            if is_dark(frame):
                consecutive_dark_frames += 1
            else:
                consecutive_dark_frames = 0

            camera_active = (consecutive_dark_frames < CONSECUTIVE_DARK_FRAMES_REQUIRED)

            if camera_active != last_camera_active:
                status_str = "ACTIVE" if camera_active else "INACTIVE (DARK)"
                logger.info(f"Camera status changed to {status_str}")

            rhino_detected = False
            rhino_count = 0

            if camera_active:
                try:
                    results_gen = model(frame, stream=True, verbose=False, half=half_precision)
                    for results in results_gen:
                        for box in results.boxes:
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = results.names[class_id]
                            if class_name == "rhino" and conf >= RHINO_CONFIDENCE_THRESHOLD:
                                rhino_detected = True
                                rhino_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), RHINO_BOX_COLOR, 2)
                                cv2.putText(frame, f"Rhino {conf:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RHINO_BOX_COLOR, 2)
                                logger.info(f"Rhino detected with {conf*100:.2f}% confidence at [{x1},{y1},{x2},{y2}].")
                except Exception as e:
                    logger.error(f"YOLO inference error: {e}", exc_info=True)
                    rhino_detected = False
            else:
                logger.info("Camera inactive; skipping detection.")
                rhino_detected = False

            # Update server if status changed or after interval
            if (rhino_detected != last_rhino_detected or
                camera_active != last_camera_active or
                (current_time - last_server_update_time) >= MIN_SERVER_UPDATE_INTERVAL):

                logger.info("Updating server with current status.")
                update_status_on_server(rhino_detected, camera_active)
                last_rhino_detected = rhino_detected
                last_camera_active = camera_active
                last_server_update_time = current_time

            # Display info on frame
            status_text = "Camera Active" if camera_active else "Camera Inactive (Dark)"
            detection_text = f"Rhino Detected ({rhino_count})" if rhino_detected else "No Rhino"
            status_color = (0, 255, 0) if camera_active else (0, 0, 255)

            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(frame, detection_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Rhino Detection Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting program by user request.")
                break

    finally:
        camera_feed.stop()
        cv2.destroyAllWindows()
        logger.info("Application terminated cleanly.")

if __name__ == "__main__":
    main()
