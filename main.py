import numpy as np
from pyquaternion import Quaternion

from intersect import find_intersection

center_lat, center_lon = 37.881684868735185, -121.91466287568684
radius = 5  # Miles
heading = Quaternion(axis=[1.0, 0.0, 0.0], angle=-np.pi * 127 / 128)
verbose = False  # Enable graphical output

print(find_intersection(center_lat, center_lon, radius, heading, verbose))
