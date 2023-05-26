from src.utils import Node, heuristic
from src.grid import Grid
from heapdict import heapdict
import numpy as np
import math

class AStar:
    """
    Does AStar path-planning on a given terrain with no updates/replanning.
    Path-planning order of operations is:
    > terrain = np.array()
    > planner = AStar(terrain)
    > path = planner.get_path(start_pos, goal_pos)
    > path_array = planner.get_plt_arrays() if you want the path as numpy array
    """
    def __init__(self, terrain: np.array):
        self.terrain = terrain
        self.grid = Grid(terrain)
        self.path = []
        self.start = None
        self.goal = None

    def calc_path(self):
        """
        Creates a path from start to goal and assigns the path to self.path.
        Path is a list of positions [(x_start, y_start), (x1, y1), ..., (x_goal, y_goal)]
        :param start: Node object start
        :param goal: Node object goal
        """
        start = self.start
        goal = self.goal
        self.init_h()

        open_list = heapdict()  # Used as a PriorityQueue with update, insert, peak, and remove features
        closed_list = set()
        open_list[start] = start.h
        parents = {start: None}  # map child to parent

        path = []
        while open_list:
            curr_node = open_list.popitem()[0]
            closed_list.add(curr_node)
            if curr_node == goal:
                while curr_node is not None:
                    path.append(curr_node.pos)
                    curr_node = parents[curr_node]
                path.reverse()
                break
            for child_pos, edge in curr_node.edges.items():
                child = self.grid.nodes[child_pos]
                if child in closed_list:
                    continue
                pos_g = curr_node.g + edge.cost
                f = pos_g + child.h
                if child in open_list and open_list[child] < f:
                    continue
                else:
                    child.g = pos_g
                    open_list[child] = f
                    parents[child] = curr_node
        self.path = path
        return

    def init_h(self):
        """
        Create h values for all nodes in the grid
        :param goal: Node object goal
        """
        for node in self.grid.nodes.values():
            node.h = heuristic(node, self.goal)
        return

    def get_path(self, start_pos: (int, int), goal_pos: (int, int)):
        """
        Function for external use to calculate path from start to goal position
        :param start_pos: tuple (int, int) starting position
        :param goal_pos: tuple (int, int) ending position
        :return: path as list of positions from start to goal
        """
        self.start = self.grid.nodes[start_pos]
        self.goal = self.grid.nodes[goal_pos]
        self.calc_path()
        return self.path

    def get_plt_arrays(self) -> np.array:
        """
        Only use after path has already been calculated through either calc_path or get_path
        :return: np.array of path where arr[:, i] is the x,y,z position at the i-th step of the path
        """
        arr = np.empty((3, len(self.path)))
        for i, pos in enumerate(self.path):
            x, y = pos
            z = self.grid.nodes[pos].height
            arr[:, i] = [x, y, z]
        return arr


class AStarUpdate:
    def __init__(self, known_map: np.array, true_map: np.array,
                 update_step: int = 1, scan_radius: int = 1, err_threshold: float = math.inf):
        self.known_map = known_map
        self.true_map = true_map
        self.known_grid = Grid(known_map)
        self.true_grid = Grid(true_map)
        self.start = None
        self.goal = None
        self.path = []
        self.update_step = update_step  # how many steps taken before updating path, 0 means not updated based on step
        self.scan_radius = scan_radius  # how many nodes around current node should be updated from true map
        self.err_thr = err_threshold  # how much difference between updated terrain and terrain pathed on before update

    def scan_terrain(self) -> (dict, float):
        diff_range = [diff for diff in range(-self.scan_radius, self.scan_radius+1)]
        x, y = self.start.pos

        changed_edges = {}
        for x_diff in diff_range:
            for y_diff in diff_range:
                x_n = x - x_diff
                y_n = y - y_diff
                if not ((0 <= x_n < self.true_grid.size) and (0 <= y_n < self.true_grid.size)):
                    # if the neighbor is outside of the original grid, skip
                    continue
                known_node = self.known_grid.nodes[(x_n, y_n)]
                true_node = self.true_grid.nodes[(x_n, y_n)]
                if known_node.height != true_node.height:
                    known_node.height = true_node.height
                    for edge in known_node.edges.values():
                        if edge not in changed_edges:
                            changed_edges[edge] = edge.cost
                        edge.update_cost()

        total_diff = 0.0

        # calculate MSE
        for edge, c_old in changed_edges.items():
            total_diff += (edge.cost - c_old) ** 2
        total_diff /= max(1.0, len(changed_edges))

        return changed_edges, total_diff

    def calc_path(self):
        self.scan_terrain()
        path = []
        latest_planner = AStar(self.known_map)
        latest_plan = latest_planner.get_path(self.start.pos, self.goal.pos)

        total_error = 0.0
        steps = 0
        while self.start != self.goal:
            if len(latest_plan) == 0:
                path = []
                break
            path.append(self.start.pos)
            next_node = self.known_grid.nodes[latest_plan[1]]
            self.start = next_node
            latest_plan = latest_plan[1:]

            if self.start != self.goal:
                _, new_error = self.scan_terrain()
                total_error += new_error
                enough_err = total_error > self.err_thr
                update_time = self.update_step > 0 and steps % self.update_step == 0
                if enough_err or update_time:
                    total_error = 0.0
                    latest_planner = AStar(self.known_grid.to_np())
                    latest_plan = latest_planner.get_path(self.start.pos, self.goal.pos)
            steps += 1
        self.path = path
        return

    def get_path(self, start_pos: (int, int), goal_pos: (int, int)):
        """
        Function for external use to calculate path from start to goal position
        :param start_pos: tuple (int, int) starting position
        :param goal_pos: tuple (int, int) ending position
        :return: path as list of positions from start to goal
        """
        self.start = self.known_grid.nodes[start_pos]
        self.goal = self.known_grid.nodes[goal_pos]
        self.calc_path()
        return self.path

    def get_plt_arrays(self) -> np.array:
        arr = np.empty((3, len(self.path)))
        for i, pos in enumerate(self.path):
            x, y = pos
            z = self.true_grid.nodes[pos].height
            arr[:, i] = [x, y, z]
        return arr

