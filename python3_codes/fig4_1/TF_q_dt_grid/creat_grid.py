import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
class Build_grid():
    def __init__(self,mx,my,start=(0,0)):
        self.mx=mx
        self.my=my
        self.rat=start
        self.target=(mx-1,my-1)
        self.grid=self.grid_zero_one()

    def grid_zero_one(self):
        grid = [[0 for x in range(self.mx)] for y in range(self.my)]
        dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0]
        cx = random.randint(0, self.mx - 1); cy = random.randint(0, self.my - 1)
        grid[cy][cx] = 1; stack = [(cx, cy, 0)]

        while len(stack) > 0:
            (cx, cy, cd) = stack[-1]
            if len(stack) > 2:
                if cd != stack[-2][2]: dirRange = [cd]
                else: dirRange = range(4)
            else: dirRange = range(4)
            nlst = []
            for i in dirRange:
                nx = cx + dx[i]; ny = cy + dy[i]
                if nx >= 0 and nx < self.mx and ny >= 0 and ny < self.my:
                    if grid[ny][nx] == 0:
                        ctr = 0
                        for j in range(4):
                            ex = nx + dx[j]; ey = ny + dy[j]
                            if ex >= 0 and ex < self.mx and ey >= 0 and ey < self.my:
                                if grid[ey][ex] == 1: ctr += 1
                        if ctr == 1: nlst.append(i)

            if len(nlst) > 0:
                ir = nlst[random.randint(0, len(nlst) - 1)]
                cx += dx[ir]; cy += dy[ir]; grid[cy][cx] = 1
                stack.append((cx, cy, ir))
            else: stack.pop()
        grid=np.asmatrix(grid)
        return grid

    def draw(self):
        fig = plt.figure(frameon=False)
        canvas = np.copy(self.grid)
        for r in range(self.mx):
            for c in range(self.my):
                if canvas[r,c] == 0.0:
                    canvas[r,c] = 1.0
                elif canvas[r,c]<0.0:
                    canvas[r,c] = 4.0
                else:
                    canvas[r,c] = 0.0
        canvas[self.target[0], self.target[1]]= 3.0 # cheese cell
        canvas[self.rat[0], self.rat[1]]= 2
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
