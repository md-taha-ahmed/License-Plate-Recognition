from PIL import Image
import cv2
import torch
import math
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

# Load YOLO models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

prev_frame_time = 0
new_frame_time = 0
IP_Webcam = False

# Load allowed plates from 'allowed.txt'
def load_allowed_plates(file_path="allowed.txt"):
    if not os.path.exists(file_path):
        print("Error: allowed.txt not found!")
        return set()
    with open(file_path, "r") as f:
        return set(line.strip() for line in f.readlines())

allowed_plates = load_allowed_plates()

# Initialize video source
if IP_Webcam is True:
    vid = cv2.VideoCapture('http://192.168.32.34:8080/videofeed')  # IP Webcam
else:
    vid = cv2.VideoCapture(0)  # Default webcam

if not vid.isOpened():
    print("Error: Unable to access the video source.")
    exit()

while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        print("Error: Could not read frame from video source.")
        break

    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()

    for plate in list_plates:
        flag = 0
        x = int(plate[0])  # xmin
        y = int(plate[1])  # ymin
        w = int(plate[2] - plate[0])  # xmax - xmin
        h = int(plate[3] - plate[1])  # ymax - ymin

        crop_img = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (int(plate[0]), int(plate[1])), (int(plate[2]), int(plate[3])), color=(0, 0, 225), thickness=2)

        for cc in range(0, 2):
            for ct in range(0, 2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    status = "Allowed" if lp in allowed_plates else "Not Allowed"
                    color = (0, 255, 0) if lp in allowed_plates else (0, 0, 255)

                    # Display license plate and its status
                    cv2.putText(frame, f"{lp} - {status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    flag = 1
                    break
            if flag == 1:
                break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)

    # Display FPS on the frame
    cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
