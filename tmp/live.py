import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from dominant_color import detect_objects_on_image

# Load your model
model = YOLO("best-cloth.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR frame to RGB PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect objects using your logic
    boxes = detect_objects_on_image(model, pil_img)
    
    for box in boxes:
        x1, y1, x2, y2, label, prob = box[:6]
        color_hex = box[6] if len(box) > 6 else None
        department = box[7] if len(box) > 7 else None

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label_text = f"{label}"
        if department:
            label_text = label_text + f" - {department}"  # override with department name

        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # Show frame in native window
    cv2.imshow("Live Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
