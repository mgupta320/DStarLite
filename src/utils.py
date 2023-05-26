import math


class Node:
    """
    This class implements a Node which represents a point in space with its position (x,y) and height. Each pixel
    in the terrain map is transformed into a node
    """
    def __init__(self, pos: (int, int), height: float, h: float = 0.0, g: float = 0.0,  rhs: float = 0.0):
        self.pos = pos
        self.height = height
        self.edges = {}  # this dictionary maps a given Node to an Edge that connects them
        self.h = h  # holds heuristic for node
        self.g = g  # cost of pathing to this node
        self.rhs = rhs  # Used in D* Lite algorithm

    def __eq__(self, other):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)

    def __str__(self):
        return f"Node at pos {self.pos}"


def heuristic(node1: Node, node2: Node) -> float:
    """
    This functions calculates euclidean distance between two different Node Objects
    :param node1: Node object
    :param node2: Node object
    :return: float Euclidean Distance
    """
    x1, y1 = node1.pos
    z1 = node1.height
    x2, y2 = node2.pos
    z2 = node2.height
    h = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)  # pythagorean theorem in 3D
    return h


class Edge:
    """
    This class implements an undirected edge which connects two Nodes and holds onto a cost
    """
    def __init__(self, u: Node, v: Node, cost: float = 0.0):
        self.u = u
        self.v = v
        self.cost = cost  # cost of traversing this edge

    def update_cost(self):
        self.cost = heuristic(self.u, self.v)  # update cost using heuristic function

    def __hash__(self):
        return hash((self.u, self.v))

    def __str__(self):
        return f"Edge connecting {self.u} and {self.v}"

