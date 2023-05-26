from src.utils import *
from src.grid import *
from src.astar import *
from src.dsl import *
import numpy as np
import math


# def sin(x, y):
#     return np.sin(x) + np.sin(y)
#
#
# lin_x = np.linspace(0, 100, 100, endpoint=False)
# lin_y = np.linspace(0, 100, 100, endpoint=False)
# xx, yy = np.meshgrid(lin_x, lin_y)
# terrain_true = 100 * sin(xx, yy)
# terrain_est = np.ones((100, 100))
#
# path_planner = DStarLite(terrain_est, terrain_true)
# path = path_planner.calc_path((10, 10), (90, 90))
# print(path)

terrain_est = np.ones((100, 100))
terrain_grid = Grid(terrain_est)
print(terrain_grid)
