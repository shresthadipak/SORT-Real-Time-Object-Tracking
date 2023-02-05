import cv2


cap = cv2.VideoCapture('videos/los_angeles.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break


    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    
