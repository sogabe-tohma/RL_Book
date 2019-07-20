import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
import time

np.random.seed(1234)
GAMMA = 0.9
LEARNING_RATE = 0.001
SMALL_ENOUGH = 0.001
fig = plt.figure(figsize=(11,9),facecolor='white')

ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

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



def random_action(a, eps=0.1):
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid, policy):
  s = (3, 0)
  grid.set_state(s)
  a = random_action(policy[s])
  states_actions_rewards = [(s, a, 0)]
  while True:
    r = grid.move(a)
    s = grid.current_state()
    if grid.game_over():
      states_actions_rewards.append((s, None, r))
      break
    else:
      a = random_action(policy[s]) 
      states_actions_rewards.append((s, a, r))
  G = 0
  states_actions_returns = []
  first = True
  for s, a, r in reversed(states_actions_rewards):
    if first:
      first = False
    else:
      states_actions_returns.append((s, a, G))
    G = r + GAMMA*G
  states_actions_returns.reverse() 
  return states_actions_returns




if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print ("rewards:")
  print_values(grid.rewards, grid)

  policy = {
    (0, 0): 'R',
    (1, 0): 'U',
    (1, 1): 'U',
    (1, 2): 'L',
    (2, 0): 'R',
    (2, 2): 'L',
    (3, 0): 'U',
    (3, 1): 'U',
    (3, 2): 'L',
  }

  # initialize theta
  # our model is V_hat = theta.dot(x)
  # where x = [row, col, row*col, 1] - 1 for bias term
  theta = np.random.randn(4) / 2
  print ('theta is intial', theta)
  def s2x(s):
    return np.array([s[0] , s[1] , s[0]*s[1], 1])

  Q = {}
  returns = {} # dictionary of state -> list of returns we've received
  states = grid.all_states()
  for s in states:
     if s in grid.actions: # not a terminal state
       Q[s] = {}
       for a in ALL_POSSIBLE_ACTIONS:
         Q[s][a] = 0
         returns[(s,a)] = []
     else:
       # terminal state or state we can't otherwise get to
       pass

  # repeat until convergence
  deltas = []
  t = 1.0
  for it in range(2000):
    if it % 100 == 0:
      t += 0.01
    alpha = LEARNING_RATE
    # generate an episode using pi
    biggest_change = 0

    states_and_returns = play_game(grid, policy)
    seen_state_action_pairs = set()
    for s,a, G in states_and_returns:
      # check if we have already seen s
      # called "first-visit" MC policy evaluation
       sa = (s, a)
       if sa not in seen_state_action_pairs:
         old_q = Q[s][a]
         returns[sa].append(G)
         Q[s][a] = np.mean(returns[sa])
         G1 = max(Q[s]['U'],Q[s]['D'],Q[s]['R'],Q[s]['L'])
         old_theta = theta.copy()
         x = s2x(s)
         V_hat = theta.dot(x)
         theta += alpha*(G1 - V_hat)*x
         biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
         seen_state_action_pairs.add(sa)
    deltas.append(biggest_change)



  plt.plot(deltas)
  plt.ylim([0,1])
  plt.show()
  print ('final theta', theta)
  # obtain predicted values
  V = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      V[s] = theta.dot(s2x(s))
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0
  print ("values:")
  print_values(V, grid)
  print ("policy:")
  print_policy(policy, grid)
