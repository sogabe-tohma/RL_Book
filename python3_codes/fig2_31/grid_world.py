
import numpy as np


class Grid: 
  def __init__(self, width, height, start):
    self.width = width
    self.height = height
    self.i = start[0]
    self.j = start[1]

  def set(self, rewards, actions):
    self.rewards = rewards
    self.actions = actions

  def set_state(self, s):
    self.i = s[0]
    self.j = s[1]

  def current_state(self):
    return (self.i, self.j)

  def is_terminal(self, s):
    return s not in self.actions

  def move(self, action):
    if action in self.actions[(self.i, self.j)]:
      if action == 'R':
        self.i += 1
      elif action == 'L':
        self.i -= 1
      elif action == 'U':
        self.j += 1
      elif action == 'D':
        self.j -= 1
    return self.rewards.get((self.i, self.j), 0)

  def undo_move(self, action):
    if action == 'U':
      self.i -= 1
    elif action == 'D':
      self.i += 1
    elif action == 'R':
      self.j -= 1
    elif action == 'L':
      self.j += 1
    assert(self.current_state() in self.all_states())

  def game_over(self):
    return (self.i, self.j) not in self.actions

  def all_states(self):
    return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_grid():
  g = Grid(4, 3, (3, 1))
  rewards = {(0, 2): 100, (0, 1): -100}
  actions = {
    (3, 2): ('D', 'L'),
    (2, 2): ('L', 'R'),
    (3, 1): ('U', 'D'),
    (1, 1): ('U', 'D', 'L'),
    (1, 2): ('D', 'L','R'),
    (0, 0): ('U', 'R'),
    (1, 0): ('L', 'R','U'),
    (2, 0): ('L', 'R'),
    (3, 0): ('L', 'U'),
  }
  g.set(rewards, actions)
  return g


def negative_grid(step_cost=-20):
  g = standard_grid()
  g.rewards.update({
    (3, 2): step_cost,
    (2, 2): step_cost,
    (3, 1): step_cost,
    (1, 1): step_cost,
    (1, 2): step_cost,
    (0, 0): step_cost,
    (1, 0): step_cost,
    (2, 0): step_cost,
    (3, 0): step_cost,
  })
  return g
