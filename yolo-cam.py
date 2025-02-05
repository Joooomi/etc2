import cv2
import numpy as np
import torch
from qai_hub_models.models.yolov7.yolov7 import YoloV7
from qai_hub_models.core.utils import load_image, preprocess_image, postprocess


GST_PIPELINE = (
    "qtiqmmfsrc name=qmmf ! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink"
)


model = YoloV7(model_id="yolov7", device="cuda" if torch.cuda.is_available() else "cpu")
model.setup()


cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    
    input_img = preprocess_image(frame, model.input_shape)

    
    preds = model(input_img)

   
    detections = postprocess(preds, model.input_shape, frame.shape)

    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f"{model.class_names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow("YOLOv7 Camera", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
