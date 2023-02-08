from kalmanFilter import KalmanFilter
from tracker_utils import convert_bbox_to_z, convert_x_to_bbox
import numpy as np


class KalmanTrack:
    '''
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    '''

    def __init__(self, initial_state):
        '''
        Initialises a tracked object according to initial bounding box.
        Initial state: single detection in form of bounding box [x_min, Y_min, X_max, Y_max]
        '''
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Transition Matrix
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        # Transformtion matrix(observation to state)
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:, 2:] *= 10.  # observation error covariance
        self.kf.P[4:, 4:] *= 1000.  # initial velocity error covariance
        self.kf.P *= 10.  # initial location error covariance
        self.kf.Q[-1, -1] *= 0.01  # process noise
        self.kf.Q[4:, 4:] *= 0.01  # process noise
        self.kf.x[:4] = convert_x_to_bbox(initial_state)  # initialize KalmanFilter state

    def project(self):
        """
        :return: (ndarray) The KalmanFilter estimated object state in [x1,x2,y1,y2] format
        """
        return convert_bbox_to_z(self.kf.x)

    def update(self, new_detection):
        """
        Updates track with new observation and returns itself after update
        :param new_detection: (np.ndarray) new observation in format [x1,x2,y1,y2]
        :return: KalmanTrack object class (itself after update)
        """
        self.kf.update(convert_x_to_bbox(new_detection))
        return self

    def predict(self):
        """
        :return: ndarray) The KalmanFilter estimated new object state in [x1,x2,y1,y2] format
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()

        return self.project()   

        