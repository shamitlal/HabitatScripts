import habitat_sim
import habitat
from habitat.config.default import get_config

import cv2

import random
#%matplotlib inline
import matplotlib.pyplot as plt
import time
import numpy as np
import ipdb
st = ipdb.set_trace
import os 
import sys
import pickle
import json


class AutomatedMultiview():
    def __init__(self):
        config = self.make_cfg()
        self.env = habitat.Env(config=config, dataset=None)
        self.run()

    def run(self):
        pts = self.get_navigable_points()
        st()
        aa=1

    def make_cfg(self):
        config = get_config("/hdd/shamit/habitat/habitat-lab/configs/tasks/pointnav.yaml")
        return config

    def get_navigable_points(self):
        navigable_points = np.array([0,0,0])
        for i in range(10000):
            navigable_points = np.vstack((navigable_points,self.env.sim.sample_navigable_point()))
        return navigable_points


if __name__ == '__main__':
    AutomatedMultiview()
