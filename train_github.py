import cv2
import math
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Make sure sort.py is in your directory

# ======================
# CONFIGURATION SETTINGS
# ======================
MODEL_PATH = "../Yolo_weights/yolov8m.pt"
VIDEO_PATH = "videos/train.mp4"
MASK_PATH = "videos/mask.png"

# Virtual line coordinates [x1, y1, x2, y2]
LEFT_LINE = [150, 638, 367, 478]
RIGHT_LINE = [918, 546, 1266, 513]

# Detection parameters
CONFIDENCE_THRESHOLD = 0.4
TRACKER_MAX_AGE = 50
TRACKER_MIN_HITS = 1
TRACKER_IOU_THRESH = 0.3


# ======================
# UTILITY FUNCTIONS
# ======================
def get_side(px: int, py: int, line: list) -> float:
    """
    Determines which side of a line a point is on
    Returns:
        >0 for one side, <0 for the other
    """
    x1, y1, x2, y2 = line
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def draw_ui(frame: np.ndarray, entry_count: int, exit_count: int) -> None:
    """Draws UI elements on frame"""
    # Draw counting lines
    cv2.line(frame, LEFT_LINE[:2], LEFT_LINE[2:], (0, 0, 255), 5)
    cv2.line(frame, RIGHT_LINE[:2], RIGHT_LINE[2:], (0, 0, 255), 5)

    # Draw counters
    cv2.putText(frame, f'Entry: {entry_count}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    cv2.putText(frame, f'Exit: {exit_count}', (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# ======================
# MAIN PROCESSING
# ======================
def main():
    # Initialize model and resources
    model = YOLO(MODEL_PATH).to("cuda")
    cap = cv2.VideoCapture(VIDEO_PATH)
    mask = cv2.imread(MASK_PATH)
    tracker = Sort(max_age=TRACKER_MAX_AGE,
                   min_hits=TRACKER_MIN_HITS,
                   iou_threshold=TRACKER_IOU_THRESH)

    # Tracking data
    previous_centers = {}
    entry_ids = set()
    exit_ids = set()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Apply ROI mask
        masked_frame = cv2.bitwise_and(frame, mask)

        # Run detection
        detections = np.empty((0, 5))
        results = model(masked_frame, stream=True, classes=[0])  # Class 0 = person

        for r in results:
            for box in r.boxes:
                if box.conf.item() > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections = np.vstack([detections, [x1, y1, x2, y2, box.conf.item()]])

        # Update tracker
        tracked_objects = tracker.update(detections)

        # Process tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw bounding box and center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Check for line crossings
            if obj_id in previous_centers:
                prev_cx, prev_cy = previous_centers[obj_id]

                # Check left line crossing
                prev_side = get_side(prev_cx, prev_cy, LEFT_LINE)
                curr_side = get_side(cx, cy, LEFT_LINE)
                if prev_side * curr_side < 0:  # Sign change = line crossed
                    if curr_side > 0 and obj_id not in entry_ids:
                        entry_ids.add(obj_id)
                    elif curr_side < 0 and obj_id not in exit_ids:
                        exit_ids.add(obj_id)

                # Check right line crossing
                prev_side = get_side(prev_cx, prev_cy, RIGHT_LINE)
                curr_side = get_side(cx, cy, RIGHT_LINE)
                if prev_side * curr_side < 0:
                    if curr_side < 0 and obj_id not in entry_ids:
                        entry_ids.add(obj_id)
                    elif curr_side > 0 and obj_id not in exit_ids:
                        exit_ids.add(obj_id)

            # Update position history
            previous_centers[obj_id] = (cx, cy)

        # Update UI
        draw_ui(frame, len(entry_ids), len(exit_ids))

        # Display
        cv2.imshow("Railway Passenger Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()