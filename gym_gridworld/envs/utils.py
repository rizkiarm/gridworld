import numpy as np
from collections import namedtuple
from skimage.draw import random_shapes

simple_map = \
[
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,1,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

unsolvable_map = \
[
    [1,1,1,1,1],
    [1,0,1,0,1],
    [1,1,1,1,1],
]

VonNeumannMotion = namedtuple('VonNeumannMotion', 
                              ['up', 'down', 'left', 'right'], 
                              defaults=[[-1, 0], [1, 0], [0, -1], [0, 1]])

# Source: https://github.com/zuoxingdong/mazelab/blob/master/mazelab/generators/random_maze.py
def random_maze(width=81, height=51, complexity=.75, density=.75):
    r"""Generate a random maze array. 
    
    It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
    is ``1`` and for free space is ``0``. 
    
    Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = np.random.randint(0, shape[1]//2 + 1) * 2, np.random.randint(0, shape[0]//2 + 1) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[np.random.randint(0, len(neighbours))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
                    
    return Z.astype(int)

# Source: https://github.com/zuoxingdong/mazelab/blob/master/mazelab/generators/random_shape_maze.py
def random_shape_maze(width, height, max_shapes, max_size, allow_overlap, shape=None):
    x, _ = random_shapes([height, width], max_shapes, max_size=max_size, multichannel=False, shape=shape, allow_overlap=allow_overlap)
    
    x[x == 255] = 0
    x[np.nonzero(x)] = 1
    
    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1
    
    return x

def solvable_map(f, *args, **kwargs):
    m = unsolvable_map
    while not solvable(m):
        m = f(*args, **kwargs)
    return m    

def flood_fill(m, pos):
    x, y = pos
    if m[x,y] == 1:
        return
    m[x,y] = 1
    for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
        new_pos = [x+dx, y+dy]
        flood_fill(m, new_pos)

def solvable(m):
    m = np.array(m)
    m = m.copy()
    for i in range(2):
        empty = list(zip(*np.where(m == 0)))
        if len(empty) == 0:
            return True
        flood_fill(m, empty[0])
    return False