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
        self.kf = KalmanFilter(dim_x = 7, dim_z = 4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0]
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1] 
                            ]) 
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]
                            ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. # give high uncertanity to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox. 
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

    

class SORT():

    def __init__(self):
        pass
            
    def obj_detector(self, img, draw=False):
        detect = objectDetector()
        img, bboxes = detect.object_detect(img)

        return img, bboxes