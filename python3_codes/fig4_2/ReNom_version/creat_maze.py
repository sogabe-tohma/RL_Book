import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
class Build_maze():
    def __init__(self,mx,my,rat=(0,0)):
        self.mx=mx
        self.my=my
        self.rat=rat
        self.target=(mx-1,my-1)
        self.maze=self.maze_zero_one()
    def maze_zero_one(self):
        maze = [[0 for x in range(self.mx)] for y in range(self.my)]
        dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
        cx = random.randint(0, self.mx - 1); cy = random.randint(0, self.my - 1)
        maze[cy][cx] = 1; stack = [(cx, cy, 0)] # stack element: (x, y, direction)
        
        while len(stack) > 0:
            (cx, cy, cd) = stack[-1]
            # to prevent zigzags:
            # if changed direction in the last move then cannot change again
            if len(stack) > 2:
                if cd != stack[-2][2]: dirRange = [cd]
                else: dirRange = range(4)
            else: dirRange = range(4)

            # find a new cell to add
            nlst = [] # list of available neighbors
            for i in dirRange:
                nx = cx + dx[i]; ny = cy + dy[i]
                if nx >= 0 and nx < self.mx and ny >= 0 and ny < self.my:
                    if maze[ny][nx] == 0:
                        ctr = 0 # of occupied neighbors must be 1
                        for j in range(4):
                            ex = nx + dx[j]; ey = ny + dy[j]
                            if ex >= 0 and ex < self.mx and ey >= 0 and ey < self.my:
                                if maze[ey][ex] == 1: ctr += 1
                        if ctr == 1: nlst.append(i)

            # if 1 or more neighbors available then randomly select one and move
            if len(nlst) > 0:
                ir = nlst[random.randint(0, len(nlst) - 1)]
                cx += dx[ir]; cy += dy[ir]; maze[cy][cx] = 1
                stack.append((cx, cy, ir))
            else: stack.pop()
        maze=np.asmatrix(maze)
        return maze

    def draw(self):
        fig = plt.figure(frameon=False)
        canvas = np.copy(self.maze)
        for r in range(self.mx):
            for c in range(self.my):
                if canvas[r,c] == 0.0:
                    canvas[r,c] = 1.0
                elif canvas[r,c]<0.0:
                    canvas[r,c] = 4.0
                else:
                    canvas[r,c] = 0.0
        canvas[self.target[0], self.target[1]]= 3.0 # cheese cell
        canvas[self.rat[0], self.rat[1]]= 2  # rat cell
        cmap = colors.ListedColormap(['white','black','green','b','red','m','yellow'])
        bounds = [0,1,2,3,4,5,6]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax = fig.gca()
        ax.set_xticks(np.arange(-0.5, self.mx, 1))
        ax.set_yticks(np.arange(-0.5, self.my, 1))
        ax.grid(linestyle="-", linewidth=1, color='black')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        img=plt.imshow(canvas, cmap=cmap, norm=norm)
        return img