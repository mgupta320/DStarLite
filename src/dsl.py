import math
from src.utils import Node, heuristic
from src.grid import Grid
from heapdict import heapdict
import numpy as np


class DStarLite:
    """
    Class to implement D*-Lite Path-Planning using a "known" map which represents the map containing known information
    and a "true" map which represents the ground truth which is scanned and used to update the known map.
    The order of operations is as follows:
    > known_map, true_map = np.arrays
    > planner = DStarLite(known_map, true_map)
    > path = planner.get_path(start_pos, goal_pos)
    > path_array = planner.get_plt_arrays() if you want the path as numpy array

    Implemented using pseudocode as found in: http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf
    """
    def __init__(self, known_map: np.array, true_map: np.array,
                 update_step: int = 1, scan_radius: int = 1, err_threshold: float = math.inf):
        self.known_map = known_map
        self.true_map = true_map
        self.known_grid = Grid(known_map)
        self.true_grid = Grid(true_map)
        self.km = 0.0  # used to keep track of previous key values when repairing path
        self.U = heapdict()  # Used as a PriorityQueue with update, insert, peak, and remove features
        self.path = []
        self.start = None
        self.goal = None
        self.update_step = update_step  # how many steps taken before updating path, 0 means not updated based on step
        self.scan_radius = scan_radius  # how many nodes around current node should be updated from true map
        self.err_thr = err_threshold  # how much difference between updated terrain and terrain pathed on before update

    def calc_key(self, node: Node) -> (float, float):
        node.h = heuristic(self.start, node)  # use chosen heuristic function
        return min(node.g, node.rhs) + node.h + self.km, min(node.g, node.rhs)

    def initialize_planning(self):
        self.U.clear()  # empty the PriorityQueue
        self.km = 0.0
        for node in self.known_grid.nodes.values():
            node.rhs = math.inf
            node.g = math.inf
        self.goal.rhs = 0
        self.U[self.goal] = (heuristic(self.start, self.goal), 0)

    def update_vertex(self, node: Node):
        if node.g != node.rhs:
            self.U[node] = self.calc_key(node)  # insert the node into the PriorityQueue or update its priority
        elif node.g == node.rhs and node in self.U:
            self.U.pop(node)  # remove from the PriorityQueue

    def compute_shortest_path(self):
        while self.U.peekitem()[1] < self.calc_key(self.start) or self.start.rhs > self.start.g:
            u, k_old = self.U.peekitem()
            k_new = self.calc_key(u)
            if k_old < k_new:
                self.U[u] = k_new
            elif u.g > u.rhs:
                u.g = u.rhs
                self.U.pop(u)
                for s_pos, c in u.edges.items():
                    s = self.known_grid.nodes[s_pos]
                    if s != self.goal:
                        s.rhs = min(s.rhs, c.cost + u.g)
                        self.update_vertex(s)
            else:
                g_old = u.g
                u.g = math.inf
                for s_pos, c in u.edges.items():
                    s = self.known_grid.nodes[s_pos]
                    if s.rhs == (c.cost + g_old):
                        if s != self.goal:
                            min_s_prime = math.inf
                            for s_prime_pos, c_prime in s.edges.items():
                                s_prime = self.known_grid.nodes[s_prime_pos]
                                min_s_prime = min(min_s_prime, c_prime.cost + s_prime.g)
                            s.rhs = min_s_prime
                    self.update_vertex(s)
                if u.rhs == g_old:
                    if u != self.goal:
                        min_s_prime = math.inf
                        for s_prime_pos, c_prime in u.edges.items():
                            s_prime = self.known_grid.nodes[s_prime_pos]
                            min_s_prime = min(min_s_prime, c_prime.cost + s_prime.g)
                        u.rhs = min_s_prime
                self.update_vertex(u)

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

        # Calculate MSE
        total_diff = 0.0
        for edge, c_old in changed_edges.items():
            total_diff += (edge.cost - c_old) ** 2
        total_diff /= max(1.0, len(changed_edges))

        return changed_edges, total_diff

    def calc_path(self):
        self.scan_terrain()
        changed_edges = {}
        path = [self.start.pos]
        steps = 0
        total_error = 0.0
        step_backs = 0

        last = self.start
        self.initialize_planning()
        self.compute_shortest_path()

        while self.start != self.goal:
            if self.start.rhs == math.inf:
                self.path = path
                return

            min_succ_cost = math.inf
            min_succ = None
            for s_prime_pos, c_prime in self.start.edges.items():
                s_prime = self.known_grid.nodes[s_prime_pos]
                temp = c_prime.cost + s_prime.g
                if temp < min_succ_cost:
                    min_succ_cost = temp
                    min_succ = s_prime
            self.start = min_succ
            path.append(self.start.pos)

            new_changed_edges, new_error = self.scan_terrain()
            changed_edges.update(new_changed_edges)
            total_error += new_error

            # print(f'Step: {steps}, Loc: {self.start.pos}, Scan Error: {new_error}, Total error: {total_error}')
            enough_err = total_error > self.err_thr
            update_time = self.update_step > 0 and steps % self.update_step == 0
            if (enough_err or update_time) and len(changed_edges) > 0:
                self.km += heuristic(last, self.start)
                last = self.start
                for edge, c_old in changed_edges.items():
                    u = edge.u
                    v = edge.v
                    c_new = edge.cost
                    if c_old > c_new:
                        if u != self.goal:
                            u.rhs = min(u.rhs, c_new + v.g)
                        if v != self.goal:
                            v.rhs = min(v.rhs, c_new + u.g)
                    else:
                        if u.rhs == c_old + v.g:
                            if u != self.goal:
                                min_rhs_cost = math.inf
                                for s_prime_pos, c_prime in u.edges.items():
                                    s_prime = self.known_grid.nodes[s_prime_pos]
                                    min_rhs_cost = min(min_rhs_cost, c_prime.cost + s_prime.g)
                                u.rhs = min_rhs_cost
                        if v.rhs == c_old + u.g:
                            if v != self.goal:
                                min_rhs_cost = math.inf
                                for s_prime_pos, c_prime in v.edges.items():
                                    s_prime = self.known_grid.nodes[s_prime_pos]
                                    min_rhs_cost = min(min_rhs_cost, c_prime.cost + s_prime.g)
                                v.rhs = min_rhs_cost
                    self.update_vertex(u)
                    self.update_vertex(v)
                total_error = 0.0
                changed_edges = {}
                self.compute_shortest_path()
            if len(path) > 3 and path[-1] == path[-3]:
                step_backs += 1
            if step_backs >= 1000:
                break
            steps += 1
        self.path = path
        return

    def get_path(self, start_pos: (int, int), goal_pos: (int, int)):
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
