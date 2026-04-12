import numpy as np
import math
import time

class OneEuroFilter:
    """The 1 Euro Filter is a first-order low-pass filter with an adaptive cutoff frequency."""
    def __init__(self, freq=30, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def update(self, x):
        t = time.time()
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x

        t_e = t - self.t_prev
        
        # Filter derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        # Filter signal
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

class HandSmoothing:
    """Refined HandSmoothing using OneEuroFilter."""
    def __init__(self, freq=30, min_cutoff=0.1, beta=0.01):
        self.filters = {} # Map of LandmarkID: (OneEuroFilterX, OneEuroFilterY)
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta

    def smooth(self, landmark_id, point):
        """point should be (x, y)"""
        if landmark_id not in self.filters:
            self.filters[landmark_id] = (
                OneEuroFilter(self.freq, self.min_cutoff, self.beta),
                OneEuroFilter(self.freq, self.min_cutoff, self.beta)
            )
        
        sx = self.filters[landmark_id][0].update(point[0])
        sy = self.filters[landmark_id][1].update(point[1])
        return sx, sy

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x, y) or objects with x, y attributes."""
    try:
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
    except:
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_palm_center(landmarks, width, height):
    """Calculates the center of the palm from specific landmarks."""
    palm_points = [0, 5, 9, 13, 17]
    xs = [landmarks[i].x * width for i in palm_points]
    ys = [landmarks[i].y * height for i in palm_points]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
