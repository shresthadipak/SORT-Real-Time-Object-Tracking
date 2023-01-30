import cv2
from objectTrackingModule import objectDetector, SORT

detector = SORT()


cap = cv2.VideoCapture('videos/los_angeles.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img, bbox = detector.obj_detector(frame)

    print(bbox)

    cv2.imshow("Object Tracking", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    
