import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
import time

SA2IDX = {}
IDX = 0
np.random.seed(23245)
fig = plt.figure(figsize=(11,9),facecolor='white')
class Model:
  def __init__(self):
    self.theta = np.random.randn(25) / np.sqrt(25)

  def sa2x(self, s, a):
    return np.array([
      s[1] - 1              if a == 'U' else 0,
      s[0] - 1.5            if a == 'U' else 0,
      (s[0]*s[1] - 3)/3     if a == 'U' else 0,
      (s[1]*s[1] - 2)/2     if a == 'U' else 0,
      (s[0]*s[0] - 4.5)/4.5 if a == 'U' else 0,
      1                     if a == 'U' else 0,
      s[1] - 1              if a == 'D' else 0,
      s[0] - 1.5            if a == 'D' else 0,
      (s[1]*s[0] - 3)/3     if a == 'D' else 0,
      (s[1]*s[1] - 2)/2     if a == 'D' else 0,
      (s[0]*s[0] - 4.5)/4.5 if a == 'D' else 0,
      1                     if a == 'D' else 0,
      s[1] - 1              if a == 'L' else 0,
      s[0] - 1.5            if a == 'L' else 0,
      (s[0]*s[1] - 3)/3     if a == 'L' else 0,
      (s[1]*s[1] - 2)/2     if a == 'L' else 0,
      (s[0]*s[0] - 4.5)/4.5 if a == 'L' else 0,
      1                     if a == 'L' else 0,
      s[1] - 1              if a == 'R' else 0,
      s[0] - 1.5            if a == 'R' else 0,
      (s[0]*s[1] - 3)/3     if a == 'R' else 0,
      (s[1]*s[1] - 2)/2     if a == 'R' else 0,
      (s[0]*s[0] - 4.5)/4.5 if a == 'R' else 0,
      1                     if a == 'R' else 0,
      1
    ])
  def predict(self, s, a):
    x = self.sa2x(s, a)
    return self.theta.dot(x)

  def grad(self, s, a):
    return self.sa2x(s, a)


def getQs(model, s):
  Qs = {}
  for a in ALL_POSSIBLE_ACTIONS:
    q_sa = model.predict(s, a)
    Qs[a] = q_sa
  return Qs



def print_values(V, g):
  time.sleep(1)
  for j in reversed(range(g.height)):

    print ("'---------------------------------------")
   
    for i in range(g.width):
      v = V.get((i,j), 0)
      if v >= 0:
          print( "| ",'{:06.2f}'.format(v), end=" ")  

      else:
         print( "| ",'{:06.2f}'.format(v), end=" ") 

    print( "| ")
  print ("'---------------------------------------")
  print ("")
  print ("")



def print_policy(P, g):
  time.sleep(2)
  for j in reversed(range(g.height)):
    print ("-----------------------------")
    for i in range(g.width):
      a = P.get((i,j), ' ')
      print( "| ",'{:3s}'.format(a),end=" ") 
    print( "| ")
  print ("-----------------------------")
  print ("")
  print ("")



def max_dict(d):
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val

def random_action(a, eps=0.1):
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

GAMMA = 0.9
ALPHA = 0.01
ALL_POSSIBLE_ACTIONS = ('R', 'U', 'D', 'L')
np.random.seed(7234323)


if __name__ == '__main__':
  grid = standard_grid()
  states = grid.all_states()
  for s in states:
    SA2IDX[s] = {}
    for a in ALL_POSSIBLE_ACTIONS:
      SA2IDX[s][a] = IDX
      IDX += 1
  model = Model()
  t = 1.0
  t2 = 1.0
  deltas = []
  for it in range(100000):
    if it % 100 == 0:
      t += 0.01
      t2 += 0.01
    if it % 1000 == 0:
      alpha = ALPHA / t2
    s = (3, 0) 
    grid.set_state(s)
    Qs = getQs(model, s)
    a = max_dict(Qs)[0]
    a = random_action(a, eps=0.5/t) 
    biggest_change = 0
    while not grid.game_over():
      r = grid.move(a)
      s2 = grid.current_state()
      old_theta = model.theta.copy()
      if grid.is_terminal(s2):
        model.theta += alpha*(r - model.predict(s, a))*model.grad(s, a)
      else:
        Qs2 = getQs(model, s2)
        a2 = max_dict(Qs2)[0]
        a2 = random_action(a2, eps=0.5/t) 
        model.theta += alpha*(r + GAMMA*model.predict(s2, a2) - model.predict(s, a))*model.grad(s, a)
        s = s2
        a = a2

      biggest_change = max(biggest_change, np.abs(model.theta - old_theta).sum())
    deltas.append(biggest_change)

  plt.plot(deltas,color = 'blue')

  plt.show()

  policy = {}
  V = {}
  Q = {}
  for s in grid.actions.keys():
        Qs = getQs(model, s)
        Q[s] = Qs
        a, max_q = max_dict(Qs)
        policy[s] = a
        V[s] = max_q
  print( "values:")
  print_values(V, grid)
  print ("policy:")
  print_policy(policy, grid)
