import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = r".\best.pt"
model = YOLO(model_path)

# Your class names in the correct order
class_names = ['hairnet', 'meat', 'apron', 'sausage', 'background']  # add 'gloves' if trained

# Define what you want to track (show on screen)
track_these = ['hairnet', 'gloves']  # Add any other safety item classes here

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = class_names[cls_id]
        conf = float(box.conf[0])

        if label in track_these and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Compliance Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
