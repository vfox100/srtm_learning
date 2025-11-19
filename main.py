import math

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

print("OK: Imports Successful")


def bound(center_lat, center_lon, radius):
    # MILES_IN_ONE_DEGREE_LAT = 69
    # MILES_IN_ONE_DEGREE_LON = 69.17
    lat_max = center_lat + radius / 69
    lat_min = center_lat - radius / 69
    lon_max = center_lon + radius / (69.17 * math.cos(lat_max))
    lon_min = center_lon - radius / (69.17 * math.cos(lat_min))
    return (lat_max, lat_min, lon_max, lon_min)  # Lat Max, Lat Min, Lon Max, Lon Min


print("OK: Function Declarations Successful")


center_lat, center_lon = 37.88177584303302, -121.55180623274018
radius = 5  # Miles
lat_max, lat_min, lon_max, lon_min = bound(center_lat, center_lon, radius)
geo_data = srtm.get_data()
altitude = geo_data.get_elevation(center_lat, center_lon)
assert altitude is not None
starting_point = (center_lon, center_lat, altitude)
heading = Quaternion(axis=[0.0, 1.0, 0.0], angle=-np.pi * 127 / 128)
direction_vec: tuple[float, float, float] = heading.rotate([1.0, 0.0, 0.0])  # pyright: ignore
direction_vec = (direction_vec[0], direction_vec[1], direction_vec[2] * 1600 * 69)
print(direction_vec)

# direction_vec = (-1.0, 0.0, 2500.0)
print("direction vec:", direction_vec)

SIZE = (512, 512)

# Creating DEM image
image = geo_data.get_image(
    SIZE, (lat_min, lat_max), (lon_min, lon_max), 840
)  # Size, Lat, Lon, max_height
grey_image = ImageOps.grayscale(image)
grey_image.save("dem.tif", format="TIFF")

# Create DEM array
LATS = np.linspace(lat_max, lat_min, SIZE[0])
LONS = np.linspace(lon_max, lon_min, SIZE[1])

dem = []  # Actual DEM array
line = []
print("Generating DEM:")
for i in tqdm.tqdm(range(0, SIZE[0])):
    for g in range(0, SIZE[1]):
        line.append(geo_data.get_elevation(LATS[i], LONS[g]))
    dem.append(line)
    line = []


print("OK: DEM Download Successful")


heights = []
print("Getting all heights:")

for i in tqdm.tqdm(dem):
    for g in i:
        if g not in heights and g is not None:
            # TODO: interpolate missing heights, which appear as None
            heights.append(g)

heights.sort()


print("OK: Retreving Heights Successful")


def intersection_at_height(
    ray_start: tuple[float, float, float],
    ray_direction: tuple[float, float, float],
    height: float,
) -> tuple[float, float, float] | None:
    t = (height - ray_start[2]) / (
        ray_direction[2] if ray_direction[2] != 0 else 0.0000001
    )
    if t <= 0:
        return None
    else:
        return (
            ray_start[0] + t * ray_direction[0],
            ray_start[1] + t * ray_direction[1],
            t,
        )


intersections = []
for possible_height in tqdm.tqdm(x for x in range(-450, 8500, 5)):
    inter = intersection_at_height(starting_point, direction_vec, possible_height)
    if inter is None:
        continue
    else:
        print("inter:", inter)
    inter_x, inter_y, t = inter
    actual_height = geo_data.get_elevation(inter_y, inter_x)
    if actual_height is None:
        continue
    if actual_height > possible_height:
        print(
            "[intersection] actual height:",
            actual_height,
            "possible height:",
            possible_height,
            "intersection:",
            inter,
        )
        intersections.append([(inter_x, inter_y), t])
    else:
        print(
            "[not an intersection] actual height:",
            actual_height,
            "possible height:",
            possible_height,
            "intersection:",
            inter,
        )

intersections.sort(key=lambda el: el[1])

first_intersection = intersections[0]

print(first_intersection)
# print(intersections)


print("OK: Calculated Intersection")


# heights2 = list({height for arr in dem for height in arr if height is not None})
# heights2.sort()

plotter = pv.Plotter()
image.show()
