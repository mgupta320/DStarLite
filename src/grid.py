from src.utils import Node, Edge
import numpy as np


class Grid:
    """
    Grid acts as a map to do path planning with. Initialized with np.array (needs ot be square)
    """
    def __init__(self, terrain: np.array):
        self.size = len(terrain)
        self.nodes = {}  # map positions (int, int) to Node object
        self.create_nodes(terrain)
        self.edges = {}  # maps node pairs (Node, Node) to Edge which connects them
        self.create_edges()

    def create_nodes(self, terrain: np.array):
        """
        Create and initialize nodes which make up terrain with default values
        :param terrain: numpy array which contains terrain information
        """
        for x in range(self.size):
            for y in range(self.size):
                # Initialize grid with positions to Node with height from given np array terrain
                self.nodes[(x, y)] = Node((x, y), terrain[x, y])

    def create_edges(self):
        """
        Create and initialize edges connecting nodes in grid
        :return:
        """
        for x in range(self.size):
            for y in range(self.size):
                node = self.nodes[(x, y)]
                for x_diff in [-1, 0, 1]:
                    for y_diff in [-1, 0, 1]:
                        if y_diff == 0 and x_diff == 0:  # if the "neighbor" is original node, skip
                            continue
                        x_n = x - x_diff
                        y_n = y - y_diff
                        if not ((0 <= x_n < self.size) and (0 <= y_n < self.size)):
                            # if the neighbor is outside of the original grid, skip
                            continue
                        if (x_n, y_n) not in node.edges:
                            neighbor = self.nodes[(x_n, y_n)]
                            edge = Edge(node, neighbor)
                            # add edge to edges to both nodes edge container and grid's edge container
                            node.edges[(x_n, y_n)] = edge
                            neighbor.edges[(x, y)] = edge
                            self.edges[(node, neighbor)] = edge
                            self.edges[(neighbor, node)] = edge
                            edge.update_cost()  # use edge's built in update_cost function

    def to_np(self) -> np.array:
        """
        Returns grid as an array of values
        :return:
        """
        container = np.empty((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                node = self.nodes[(x, y)]
                container[x, y] = node.height
        return container

    def __str__(self):
        return f"{self.size}x{self.size} Grid:\n {self.to_np()}"
