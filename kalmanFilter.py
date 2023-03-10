import numpy as np
from numpy import dot, zeros, eye
from numpy.linalg import inv

class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = zeros((dim_x, 1))
        self.P = eye(dim_x)
        self.Q = eye(dim_x)
        self.F = eye(dim_x)
        self.H = zeros((dim_z, dim_x))
        self.R = eye(dim_z)
        self.M = zeros((dim_z, dim_z))

        self._I = eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def predict(self):
        '''
        Predict next state using the kalman filter state propagation equations.
        '''
        self.x = dot(self.F, self.x)                            # x = Fx 
        self.P = dot(self.F, dot(self.P, self.F.T)) + self.Q    # P = FPF' + Q
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z):
        '''
        At the time step k, this update step computes the posterior mean x and covariance P
        of the system state given a new measurement z.
        '''
        # y = z - Hx (Residual between measurement and prediction)
        y = z - np.dot(self.H, self.x)
        PHT = dot(self.P, self.H.T)

        # S = HPH' + R (Project system uncertainty into measurement space)
        S = dot(self.H, PHT) + self.R

        # K = PH'S^-1  (map system uncertainty into Kalman gain)
        K = dot(PHT, inv(S))

        # x = x + Ky  (predict new x with residual scaled by the Kalman gain)
        self.x = self.x + dot(K, y)

        # P = (I-KH)P
        I_KH = self._I - dot(K, self.H)
        self.P = dot(I_KH, self.P)    