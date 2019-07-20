import numpy as np

class Grid: # Environment
  def __init__(self):
    self.width = 3
    self.height = 4
    self.i = 3
    self.j = 1
    self.start=[2,0]
    self.goal=[0,3]
    self.s_state=[self.i,self.j],np.shape(2)
    self.t_state={(1, 1),(1,2),(1,3),(1,4),(2,1),(3,1),(2,3),(2,4),(2,2),(3,2),(3,3),(3,4)}
    self.t_action=['U','R','L','D']
    self.actions ={
       (3, 3): ('R', 'L','U'),
       (3, 4): ('L','U'),
       (2, 2): ('L','U', 'D','R'),
       (2, 3): ('L','R','D','U'),
       (2, 4): ('L','U','D'),
       (3, 2) : ('U','R','L'),
       (1, 1): ( 'R','D'),
       (2, 1): ( 'R','D','U'),
       (3, 1): ( 'R','U'),
       (1, 2): ('R','L', 'D'),
       (1, 3): ('R','L', 'D'),
       }

    self.rewards= {(1, 4):1}

  def set(self, state):
    self.i = state[0]
    self.j = state[1]
  def reset(self):
    self.__init__()
    return [self.i, self.j]

  def step(self,action):
    Reward=self.move(action)
    state=[self.i,self.j]
    terminate=self.game_over()
    if terminate:
        self.reset()
    return state,Reward,terminate

  def current_state(self):
    return [self.i, self.j]

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    # check if legal move first
    if action in self.actions[(self.i, self.j)]:
      if action == 'U':
        self.i -= 1
      elif action == 'D':
        self.i += 1
      elif action == 'R':
        self.j += 1
      elif action == 'L':
        self.j -= 1
      else:
        self.i+=0
        self.j+=0
    return self.rewards.get((self.i, self.j), -0.1)

  def undo_move(self, action):
    if action == 'U':
      self.i += 1
    elif action == 'D':
      self.i -= 1
    elif action == 'R':
      self.j -= 1
    elif action == 'L':
      self.j += 1
    assert(self.current_state() in self.all_states())

  def game_over(self):
    return (self.i, self.j) not in self.actions

  def all_states(self):
    return set(self.actions.keys()+ self.rewards.keys())

  def draw_board(self):
        board = []
        for I in range(self.width):
            for J in range(self.height):
                if self.start[0]==I and self.start[1]==J:
                    board.append("S")
                elif self.goal[0]==I and self.goal[1]==J:
                    board.append("G")
                else:
                    board.append(" ")
        print(" "*15, ".....................")
        print(" "*15,"|","".join(board[0]), " |", "".join(board[1]), " |", "".join(board[2])," |","".join(board[3])," |")
        print(" "*15,"|----|----|----|----|")
        print(" "*15,"|","".join(board[4]), " |", "".join(board[5]), " |", "".join(board[6])," |","".join(board[7])," |")
        print(" "*15,"|----|----|----|----|")
        print(" "*15,"|","".join(board[8]), " |", "".join(board[9]), " |", "".join(board[10])," |","".join(board[11])," |")

        print(" "*15, "''''''''''''''''''''")
