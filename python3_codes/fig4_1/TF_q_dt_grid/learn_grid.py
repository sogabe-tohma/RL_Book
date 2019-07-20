import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

num_actions=4
class Qgrid():
    def __init__(self, grid, rat=(2,3)):
        self.rat=rat
        self.actions_dict = {
            0: ['LEFT','left'],
            1: ['UP','up'],
            2:[ 'RIGHT','right'],
            3: ['DOWN','down'],
        }
        self._grid = np.array(grid)
        nrows, ncols = self._grid.shape
        self.target = (0, 0)   # target cell where the "cheese" is
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._grid[r,c] == 1.0]
        self.N_goal=[(r,c) for r in range(nrows) for c in range(ncols) if self._grid[r,c] == -1.0][0]
        self.free_cells.remove(self.target)
        if self._grid[self.target] == 0.0:
            raise Exception("Invalid grid: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(self.rat)

    def reset(self, rat):
        self.grid = np.copy(self._grid)
        nrows, ncols = self.grid.shape
        row, col = rat
        self.grid[row, col] = 0.5
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.grid.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        if action == 0 or action ==self.actions_dict[0][0] or action ==self.actions_dict[0][1]:
            action=0
        elif action == 1 or action ==self.actions_dict[1][0] or action ==self.actions_dict[1][1]:
            action=1
        elif action ==2 or action ==self.actions_dict[2][0] or action ==self.actions_dict[2][1]:
            action=2
        elif action ==3 or action ==self.actions_dict[3][0] or action ==self.actions_dict[3][1]:
            action=3
        nrows, ncols = self.grid.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.grid[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == 0 :
                ncol -= 1
            elif action == 1:
                nrow -= 1
            elif action ==2 :
                ncol += 1
            elif action == 3:
                nrow += 1
        else:
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.grid.shape
        if rat_row == self.target[0] and rat_col == self.target[1]:
            return 1.0
        if rat_row == self.N_goal[0] and rat_col == self.N_goal[1]:
            return -1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        #act return the state reward and nextstate
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate=[]
        nrows, ncols = canvas.shape
        for r in range(nrows):
            for c in range(ncols):
                envstate.append([canvas[r,c]])
        envstate=np.reshape(envstate,[1,np.size(canvas)])
        return envstate

    def draw(self,all_states):
        fig = plt.figure(frameon=False)
        ims = []
        for states in all_states:
            canvas = np.copy(self.grid)
            nrows, ncols = self.grid.shape
            for r in range(nrows):
                for c in range(ncols):
                    if canvas[r,c] == 0.0:
                        canvas[r,c] = 1.0
                    elif canvas[r,c]<0.0:
                        canvas[r,c] = 4.0
                    else:
                        canvas[r,c] = 0.0
            canvas[self.target[0], self.target[1]]= 3.0 # cheese cell
            step=0
            for a in range(len(states)):
                for item in range(step):
                    canvas[states[item][0],states[item][1]]=6.0
                canvas[states[a][0], states[a][1]]= 2  # rat cell
                cmap = colors.ListedColormap(['white','black','green','b','red','m','yellow'])
                bounds = [0,1,2,3,4,5,6]
                norm = colors.BoundaryNorm(bounds, cmap.N)
                ax = fig.gca()
                ax.set_xticks(np.arange(-0.5, nrows, 1))
                ax.set_yticks(np.arange(-0.5, ncols, 1))
                ax.grid(linestyle="-", linewidth=1, color='black')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ims.append([plt.imshow(canvas, cmap=cmap, norm=norm)])
                step+=1
        ani=animation.ArtistAnimation(fig, ims, interval=200, blit=False, repeat_delay=200)
        ani.save('regression_anim_i.mp4', fps=4)
        plt.close()
    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.grid.shape
        if rat_row == self.target[0] and rat_col == self.target[1]:
            return 'win'

        return 'not_over'

    def draw_env(self):
        canvas = np.copy(self.grid)
        nrows, ncols = self.grid.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = 2.5
        return canvas

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.grid.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)
        if row>0 and self.grid[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.grid[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.grid[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.grid[row,col+1] == 0.0:
            actions.remove(2)

        return actions
