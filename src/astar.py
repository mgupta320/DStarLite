from src.utils import Node, heuristic
from src.grid import Grid
from heapdict import heapdict
import numpy as np


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

    def calc_path(self, start: Node, goal: Node):
        """
        Creates a path from start to goal and assigns the path to self.path.
        Path is a list of positions [(x_start, y_start), (x1, y1), ..., (x_goal, y_goal)]
        :param start: Node object start
        :param goal: Node object goal
        """
        self.init_h(goal)

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

    def init_h(self, goal: Node):
        """
        Create h values for all nodes in the grid
        :param goal: Node object goal
        """
        for node in self.grid.nodes.values():
            node.h = heuristic(node, goal)
        return

    def get_path(self, start_pos: (int, int), goal_pos: (int, int)):
        """
        Function for external use to calculate path from start to goal position
        :param start_pos: tuple (int, int) starting position
        :param goal_pos: tuple (int, int) ending position
        :return: path as list of positions from start to goal
        """
        start_node = self.grid.nodes[start_pos]
        goal_node = self.grid.nodes[goal_pos]
        self.calc_path(start_node, goal_node)
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
