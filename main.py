import math

import numpy as np
import PIL as img
import pyvista as pv
import srtm
import tqdm
from PIL import ImageOps
from pyquaternion import Quaternion
from scipy import interpolate
from vtkmodules.numpy_interface.dataset_adapter import NoneArray

# ABSOLUTE ZER0: pointing along
#  * positive x, or 90 deg East,
#  * positive y, or 0 deg North,
#  * positive z, or up


def bound(center_lat, center_lon, radius):
    # MILES_IN_ONE_DEGREE_LAT = 69
    # MILES_IN_ONE_DEGREE_LON = 69.17
    lat_max = center_lat + radius / 69
    lat_min = center_lat - radius / 69
    lon_max = center_lon + radius / (69.17 * math.cos(lat_max))
    lon_min = center_lon - radius / (69.17 * math.cos(lat_min))
    return (lat_max, lat_min, lon_max, lon_min)  # Lat Max, Lat Min, Lon Max, Lon Min


center_lat, center_lon = 37.881684868735185, -121.91466287568684
geo_data = srtm.get_data()
altitude = geo_data.get_elevation(center_lat, center_lon)
radius = 5  # Miles
lat_max, lat_min, lon_max, lon_min = bound(center_lat, center_lon, radius)
assert altitude is not None
starting_point = (center_lon, center_lat, altitude)
heading = Quaternion(axis=[0.0, 1.0, 0.0], angle=-np.pi * 127 / 128)


SIZE = (512, 512)
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

dem = np.array(dem, dtype=float)

x = np.arange(0, dem.shape[1])
y = np.arange(0, dem.shape[0])
# mask invalid values
dem = np.ma.masked_invalid(dem)
xx, yy = np.meshgrid(x, y)
# get only the valid values
x1 = xx[~dem.mask]
y1 = yy[~dem.mask]
dem_interpolated = dem[~dem.mask]

GD1 = interpolate.griddata((x1, y1), dem_interpolated.ravel(), (xx, yy), method="cubic")
# dem_interpolated = dem_interpolated.reshape(SIZE)


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
image = geo_data.get_image(
    SIZE, (lat_min, lat_max), (lon_min, lon_max), 840
)  # Size, Lat, Lon, max_height
grey_image = ImageOps.grayscale(image)
grey_image.save("dem.tif", format="TIFF")
print("OK: DEM Download Successful")
"""


plotter = pv.Plotter()
# image.show()

scale_lat = ((lat_max - lat_min) * 111132) / SIZE[
    0
]  # Number of meters per pixel latitude
scale_lon = ((lon_max - lon_min) * 111132 * math.cos(center_lat)) / SIZE[
    1
]  # number of meters per pixel longitude

points = []

for i in range(0, len(dem)):
    for g in range(0, len(dem[0])):
        points.append(
            [
                i,
                g,
                (dem[i][g] / scale_lat) if dem[i][g] is not None else 0,
            ]
        )

points = np.array(points)

print(points)

plotter = pv.Plotter()
point_cloud = pv.PolyData(points)

data = points[:, -1]
point_cloud["elevation"] = data

zcol = points[:, 2]
point_cloud = point_cloud.extract_points(np.isfinite(zcol), include_cells=False)


plotter.add_mesh(
    point_cloud.delaunay_2d(),
    scalars="elevation",
    show_edges=True,
)

sphere = pv.Sphere(
    radius=1, center=[math.floor(512 / 2), math.floor(512 / 2), altitude / scale_lat]
)
plotter.add_mesh(sphere, color="black", opacity=1.0)
# plotter.enable_eye_dome_lighting()
plotter.show()


# Use this point: [math.floor(512 / 2), math.floor(512 / 2), 40]
