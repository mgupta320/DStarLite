import math
import matplotlib.pyplot as plt
from src.utils import *
from src.grid import *
from src.astar import *
from src.dsl import *
import time


def sin(x, y, rez, scale):
    return np.sin(1.0*x*scale/rez) + np.sin(1.0*y*scale/rez)


def plot_path(ax, planner, true_map):
    arr_path = planner.get_plt_arrays()
    xx = arr_path[0, :]
    yy = arr_path[1, :]

    ax.imshow(true_map, cmap='terrain')
    ax.plot(yy, xx, 'r')
    ax.plot(yy[0], xx[0], 'r*')
    ax.plot(yy[-1], xx[-1], 'rX')
    return


size = 200
mag = 10.0
lin_x = np.linspace(0, size, size, endpoint=False)
lin_y = np.linspace(0, size, size, endpoint=False)
xx, yy = np.meshgrid(lin_x, lin_y)
terrain_true = 100.0 * sin(xx, yy, size, mag)
terrain_est = np.zeros((size, size))
known_grid = Grid(terrain_est)
true_grid = Grid(terrain_true)

fig = plt.figure()

path_planner_base = AStar(terrain_true)
start = time.time()
path_base = path_planner_base.get_path((10, 10), (size-10, size-10))
end = time.time()
elapsed = end - start
base_cost = true_grid.calc_path_cost(path_base)
print(f"AStarBaseline path cost = {base_cost} in {elapsed} sec\n")
ax1 = fig.add_subplot(221)
ax1.set_title('Baseline A*')
plot_path(ax1, path_planner_base, terrain_true)

path_planner_naive = AStar(terrain_est)
start = time.time()
path_naive = path_planner_naive.get_path((10, 10), (size-10, size-10))
end = time.time()
elapsed = end - start
naive_cost = true_grid.calc_path_cost(path_naive)
print(f"AStarNaive path cost = {naive_cost} in {elapsed} sec\n")
ax2 = fig.add_subplot(222)
ax2.set_title('Naive A*')
plot_path(ax2, path_planner_naive, terrain_true)

path_planner_update = AStarUpdate(terrain_est, terrain_true, 1, 10, math.inf)
start = time.time()
path_update = path_planner_update.get_path((10, 10), (size-10, size-10))
end = time.time()
elapsed = end - start
update_cost = true_grid.calc_path_cost(path_update)
print(f"AStarUpdate path cost = {update_cost} in {elapsed} sec\n")
ax3 = fig.add_subplot(223)
ax3.set_title('A* Update')
plot_path(ax3, path_planner_update, terrain_true)

path_planner_dsl = DStarLite(terrain_est, terrain_true, 1, 10, math.inf)
start = time.time()
path_dsl = path_planner_dsl.get_path((10, 10), (size-10, size-10))
end = time.time()
elapsed = end - start
dsl_cost = true_grid.calc_path_cost(path_dsl)
print(f"DStar path cost = {dsl_cost} in {elapsed} sec\n")
ax4 = fig.add_subplot(224)
ax4.set_title('D* Lite')
plot_path(ax4, path_planner_dsl, terrain_true)

plt.show()
