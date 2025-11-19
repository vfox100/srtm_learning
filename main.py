import math
from http.client import NotConnected

import numpy as np
import PIL as img
import pyvista as pv
import srtm
import tqdm
from PIL import ImageOps
from pyquaternion import Quaternion
from vtkmodules.numpy_interface.dataset_adapter import NoneArray

# ABSOLUTE ZER0: pointing along
#  * positive x, or 90 deg East,
#  * positive y 0 deg North
#  * positive z, or up

center_lat, center_lon = 37.88177584303302, -121.55180623274018
geo_data = srtm.get_data()
altitude = geo_data.get_elevation(center_lat, center_lon)
assert altitude is not None
starting_point = (center_lon, center_lat, altitude)
heading = Quaternion(axis=[0.0, 1.0, 0.0], angle=-np.pi * 127 / 128)


def get_intersection(
    starting_point: tuple[float, float, float], heading: Quaternion
) -> tuple[float, float, float] | None:
    direction_vec: tuple[float, float, float] = heading.rotate([1.0, 0.0, 0.0])  # pyright: ignore
    direction_vec = (direction_vec[0], direction_vec[1], direction_vec[2] * 1600 * 69)
    direction_mag_deg: float = np.sqrt(
        direction_vec[0] ** 2
        + direction_vec[1] ** 2
        + (direction_vec[2] / (1600 * 69)) ** 2
    )
    direction_mag_m: float = np.sqrt(
        (direction_vec[0] * 1600 * 69) ** 2
        + (direction_vec[1] * 1600 * 69) ** 2
        + direction_vec[2] ** 2
    )
    increment_m = 160.0
    increment_deg = increment_m / (1600.0 * 69.0)
    current_point = starting_point
    while (
        -450.0 < current_point[2] < 9000.0
        and abs(current_point[0] - starting_point[0]) < 5.0
        and abs(current_point[1] - starting_point[1]) < 5.0
    ):
        earth_elevation = geo_data.get_elevation(current_point[1], current_point[0])
        if earth_elevation is not None and earth_elevation > current_point[2]:
            return current_point
        current_point = (
            current_point[0] + increment_deg * direction_vec[0] / direction_mag_deg,
            current_point[1] + increment_deg * direction_vec[1] / direction_mag_deg,
            current_point[2] + increment_m * direction_vec[2] / direction_mag_m,
        )
    return None


print(get_intersection(starting_point, heading))

# Creating DEM image
"""
def bound(center_lat, center_lon, radius):
    # MILES_IN_ONE_DEGREE_LAT = 69
    # MILES_IN_ONE_DEGREE_LON = 69.17
    lat_max = center_lat + radius / 69.0
    lat_min = center_lat - radius / 69.0
    lon_max = center_lon + radius / (69.17 * math.cos(lat_max))
    lon_min = center_lon - radius / (69.17 * math.cos(lat_min))
    return (lat_max, lat_min, lon_max, lon_min)  # Lat Max, Lat Min, Lon Max, Lon Min

radius = 5  # Miles
lat_max, lat_min, lon_max, lon_min = bound(center_lat, center_lon, radius)
image = geo_data.get_image(
    SIZE, (lat_min, lat_max), (lon_min, lon_max), 840
)  # Size, Lat, Lon, max_height
grey_image = ImageOps.grayscale(image)
grey_image.save("dem.tif", format="TIFF")
print("OK: DEM Download Successful")
"""


plotter = pv.Plotter()
# image.show()
