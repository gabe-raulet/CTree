import numpy as np
import matplotlib.pyplot as plt
import sys

def gen_point_cloud(n, d, a, b):
    data = np.random.random(n*d)
    data = (b - a)*data + a
    return data.reshape((n,d))

def dist(p, q):
    return np.sqrt(np.dot(p-q, p-q))

class CoverNode(object):
    def __init__(self, tree, node_id, point_id, parent_id, level, covering):
        self.tree = tree
        self.node_id = node_id
        self.point_id = point_id
        self.parent_id = parent_id
        self.level = level
        self.covering = covering
        self.child_node_ids = list()
        if parent_id != -1:
            parent_node = self.get_parent_node()
            parent_node.child_node_ids.append(self.node_id)

    def __repr__(self):
        return f"CoverNode(point_id={self.point_id}, node_id={self.node_id}, parent_id={self.parent_id}, level={self.level}, covering={self.covering}, child_node_ids={self.child_node_ids})"

    def get_parent_node(self):
        return self.tree.get_node(self.parent_id)

    def get_child_nodes(self):
        for child_id in self.child_node_ids:
            yield self.tree.get_node(child_id)

    def covdist(self):
        return self.tree.covdist(self.level)

    def sepdist(self):
        return self.tree.sepdist(self.level)

    def __eq__(self, other):
        return self.node_id == other.node_id

class CoverTree(object):
    def __init__(self, points, max_radius):
        self.points = points
        self.max_radius = max_radius
        self.nodes = list()
        self.levels = dict()
        self.max_level = 5

    def covdist(self, level):
        return 2**(-level)

    def sepdist(self, level):
        if level + 1 >= self.max_level: return 0.0
        else: return 0.5*self.covdist(level)

    @classmethod
    def build_tree(cls, points):
        max_radius = max([dist(points[0,:], points[i,:]) for i in range(points.shape[0])])
        tree = cls(points, max_radius)
        tree.build_nodes()
        return tree

    def num_points(self):
        return points.shape[0]

    def num_nodes(self):
        return len(self.nodes)

    def get_node(self, node_id):
        assert node_id < self.num_nodes()
        if node_id < 0: return None
        else: return self.nodes[node_id]

    def dist(self, i, j):
        return dist(self.points[i,:], self.points[j,:]) / self.max_radius

    def get_level_nodes(self, level):
        if level in self.levels:
            for node_id in self.levels[level]:
                yield self.get_node(node_id)

    def separation(self, level):
        point_ids = [node.point_id for node in self.get_level_nodes(level)]
        m = len(point_ids)
        separation = np.inf
        for i in range(m):
            for j in range(i+1,m):
                separation = min(separation, self.dist(point_ids[i], point_ids[j]))
        return separation

    def build_node(self, point_id, parent_id, level, covering):
        node = CoverNode(tree=self, node_id=self.num_nodes(), point_id=point_id, parent_id=parent_id, level=level, covering=covering)
        self.nodes.append(node)
        if not node.level in self.levels:
            self.levels[node.level] = set()
        self.levels[node.level].add(node.node_id)
        # assert self.separation(node.level) > self.sepdist(node.level-1)
        return node

    def build_root(self):
        assert self.num_nodes() == 0
        return self.build_node(point_id=0, parent_id=-1, level=0, covering=np.arange(self.num_points()))

    def build_child_node(self, parent_node, point_id, covering):
        return self.build_node(point_id=point_id, parent_id=parent_node.node_id, level=parent_node.level+1, covering=covering)

    def build_child_nodes(self, node):
        covering = node.covering
        m = len(covering)
        coverdists = np.zeros(m, dtype=np.double)
        closest = np.zeros(m, dtype=int)
        to_build = []
        for i in range(m):
            coverdists[i] = self.dist(node.point_id, covering[i])
            if covering[i] == node.point_id:
                to_build.append(i)
        assert(len(to_build) == 1)
        k = 0
        while True:
            farthest = np.argmax(coverdists)
            if coverdists[farthest] <= self.sepdist(node.level):
                break
            to_build.append(farthest)
            k += 1
            for i in range(m):
                lastdist = coverdists[i]
                curdist = self.dist(covering[farthest], covering[i])
                if curdist <= lastdist:
                    coverdists[i] = curdist
                    closest[i] = k
        for k, j in enumerate(to_build):
            child_covering = [covering[i] for i in range(m) if closest[i] == k]
            self.build_child_node(parent_node=node, point_id=covering[j], covering=child_covering)


    def build_nodes(self):
        root_node = self.build_root()
        stack = [root_node]
        while len(stack) != 0:
            node = stack.pop()
            self.build_child_nodes(node)
            for child_node in node.get_child_nodes():
                if len(child_node.covering) > 1:
                    stack.append(child_node)

    def bfs(self):
        for l in sorted(list(self.levels.keys())):
            for node in self.get_level_nodes(l):
                yield node

    def dfs(self):
        stack = [self.nodes[0]]
        visited = set()
        while len(stack) != 0:
            u = stack.pop()
            if not u.node_id in visited:
                visited.add(u.node_id)
                yield u
                for v in u.get_child_nodes():
                    stack.append(v)

    def check_nesting(self, log=False):
        correct = True
        for u in self.bfs():
            if u.level >= self.max_level: continue
            num_nested = sum([int(bool(u.point_id == v.point_id)) for v in u.get_child_nodes()])
            if num_nested > 1:
                correct = False
                if log: print(f"[check_nesting failed] :: node {u} has {num_nested} nested children")
                else: return False
        return correct

    def check_covering(self, log=False):
        correct = True
        for u in self.bfs():
            for v in u.get_child_nodes():
                if self.dist(u.point_id, v.point_id) > u.covdist():
                    correct = False
                    if log: print(f"[check_covering failed] :: node u={u} has child v={v} with d(u,v) = {self.dist(u.point_id, v.point_id):.3f} > {u.covdist():.3f} = covdist(u)")
                    else: return False
        return correct

    def check_separation(self, log=False):
        correct = True
        for l in sorted(list(self.levels.keys())):
            sep = self.separation(l)
            if sep <= self.sepdist(l-1):
                correct = False
                if log: print(f"[check_separation failed] :: level {l} separation is {sep:.3f} but should be greater than {self.sepdist(l-1):.3f}")
                else: return False
        return correct

    def plot_points(self, indices=None, color="black", s=0.75):
        if indices is None: indices = list(range(self.num_points()))
        plt.scatter(self.points[indices,0], self.points[indices,1], color=color, s=s)

    def plot_point_names(self, indices=None, color="red", fontsize=6):
        if indices is None: indices = list(range(self.num_points()))
        for i in indices:
            pt = self.points[i,:]
            plt.text(pt[0]*(1.01), pt[1]*(1.01), i, color=color, fontsize=fontsize)

    def plot_circles(self, indices, radii, color="red", alpha=0.03, linewidth=0.2):
        assert len(indices) == len(radii)
        for i, r in zip(indices, radii):
            ball = plt.Circle(self.points[i,:], r, color=color, alpha=alpha, fill=True, linewidth=linewidth)
            plt.gca().add_patch(ball)

    def plot_line(self, i, j, linestyle="-", color="black", linewidth=0.2):
        xvals = [self.points[i,0], self.points[j,0]]
        yvals = [self.points[i,1], self.points[j,1]]
        plt.plot(xvals, yvals, marker=None, linestyle=linestyle, color=color, linewidth=linewidth)

    def plot_node(self, node):
        self.plot_points(indices=[node.point_id], color="black", s=0.75)
        self.plot_point_names(indices=[node.point_id], color="black")
        self.plot_circles(indices=[node.point_id], radii=[node.covdist()*self.max_radius], color="blue", alpha=0.03)
        indices = []
        radii = []
        for child_node in node.get_child_nodes():
            indices.append(child_node.point_id)
            radii.append(node.covdist()*0.5*self.max_radius)
            self.plot_line(node.point_id, child_node.point_id)
        self.plot_points(indices=indices, color="red", s=0.5)
        self.plot_point_names(indices=indices, color="red")
        self.plot_circles(indices=indices, radii=radii, color="red", alpha=0.05)

    def plot_level(self, level):
        for node in self.get_level_nodes(level):
            self.plot_node(node)

# np.random.seed(10)
points = gen_point_cloud(n=130, d=2, a=-1, b=1)
tree = CoverTree.build_tree(points)
tree.check_nesting(log=True)
tree.check_covering(log=True)
tree.check_separation(log=True)
for l in sorted(tree.levels.keys()):
    print(f"sep({l}) = {tree.separation(l):.3f}; sepdist({l}) = {tree.sepdist(l-1):.3f}")
