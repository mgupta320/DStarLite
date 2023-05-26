from src.utils import *
from src.grid import *
from src.astar import *
from src.dsl import *
import time


def sin(x, y):
    return np.sin(1.0*x) + np.sin(1.0*y)


size = 100
lin_x = np.linspace(0, size, size, endpoint=False)
lin_y = np.linspace(0, size, size, endpoint=False)
xx, yy = np.meshgrid(lin_x, lin_y)
terrain_true = size/2.0 * sin(xx, yy)
terrain_est = np.ones((size, size))
true_grid = Grid(terrain_true)

path_planner_base = AStar(terrain_true)
start = time.time()
path_base = path_planner_base.get_path((10, 10), (90, 90))
end = time.time()
elapsed = end - start
base_cost = true_grid.calc_path_cost(path_base)
print(f"AStarBaseline path cost = {base_cost} in {elapsed} sec")

path_planner_naive = AStar(terrain_est)
start = time.time()
path_naive = path_planner_naive.get_path((10, 10), (90, 90))
end = time.time()
elapsed = end - start
naive_cost = true_grid.calc_path_cost(path_naive)
print(f"AStarNaive path cost = {naive_cost} in {elapsed} sec")

path_planner_update = AStarUpdate(terrain_est, terrain_true, scan_radius=5)
start = time.time()
path_update = path_planner_update.get_path((10, 10), (90, 90))
end = time.time()
elapsed = end - start
update_cost = true_grid.calc_path_cost(path_update)
print(f"AStarUpdate path cost = {update_cost} in {elapsed} sec")

path_planner_dsl = DStarLite(terrain_est, terrain_true, scan_radius=5)
start = time.time()
path_dsl = path_planner_dsl.get_path((10, 10), (90, 90))
end = time.time()
elapsed = end - start
dsl_cost = true_grid.calc_path_cost(path_dsl)
print(f"DStar path cost = {dsl_cost} in {elapsed} sec")
