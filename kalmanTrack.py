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

        