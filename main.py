import cvzone
from ultralytics import YOLO
import cv2




video = r'C:\Users\Admin\Desktop\testvids/2.mp4'

cap = cv2.VideoCapture(0)
facemodel = YOLO('yolov8m.pt')


while cap.isOpened():
    rt, video = cap.read()
    video = cv2.resize(video, (1020, 720))
    mainvideo = video.copy()

    face_result = facemodel.predict(video)
    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h,w = y2-y1,x2-x1

            cvzone.cornerRect(video,[x1,y1,w,h],l=9,rt=3)
            cvzone.cornerRect(mainvideo, [x1, y1, w, h], l=9, rt=3)

            face = video[y1:y1+h,x1:x1+w]
            face = cv2.blur(face,(30,30))
            video[y1:y1+h,x1:x1+w] = face

    print(face_result['names'[0]])
    allFeeds = cvzone.stackImages([mainvideo,video],2,0.70)
    cv2.imshow('frame',mainvideo)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()



'''
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 
 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
 62: 'tv', 
 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
'''
