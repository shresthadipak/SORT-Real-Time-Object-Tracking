import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

yolov4_weights = "YOLOv4_model/yolov4.weights"
yolov4_cfg = "YOLOv4_model/yolov4.cfg"
coco_names = "YOLOv4_model/coco.names"

def convert_bbox_to_z(bbox):
    # takes a bounding box in the form [x1, y1, x2, y2] and returns in the form [x, y, s, r]
    # where x, y is the centre of the box and  si scale/area and r is aspect ratio
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2
    y = bbox[1] + h/2
    s = w * h # scale is just area
    r = w/float(h)
    
    # return [x, y, s, r]
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    #Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    #[x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    w = np.sqrt(x[2] * x[3])
    h = x[2]/w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class objectDetector():

    def __init__(self, yolov4_weights = yolov4_weights, yolov4_cfg = yolov4_cfg, coco_names = coco_names):
        self.yolov4_weights = yolov4_weights
        self.yolov4_cfg = yolov4_cfg
        self.coco_names = coco_names

        self.yolo = cv2.dnn.readNet(self.yolov4_weights, self.yolov4_cfg)
        self.layer_names = self.yolo.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.yolo.getUnconnectedOutLayers()]

        with open(self.coco_names, "r") as file:
            self.classes = [line.strip() for line in file.readlines()] 

        self.colorWhite = (255, 255, 255)   

    def object_detect(self, img, draw=True):
        height, width, channels = img.shape

        # detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        bboxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    bboxes.append([x, y, x+w, y+h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
          
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        for i, conf in zip(range(len(boxes)), confidences):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[i]
                text = label+ ' ' +str(round(conf, 2))
                if draw:
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                    cv2.circle(img, (x + w, y + h), 3, (0, 0, 255), -1)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorWhite, 1)

        return img, bboxes

class KalmanBoxTracker():

    def __init__(self, bbox):
        # initilize a tracker using initial bounding box
         
        # define constant velocity model
        pass    

    

class SORT():

    def __init__(self):
        pass
            
    def obj_detector(self, img, draw=False):
        detect = objectDetector()
        img, bboxes = detect.object_detect(img)

        return img, bboxes