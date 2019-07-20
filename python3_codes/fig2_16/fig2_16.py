import numpy as np
from grid_world import standard_grid, negative_grid
import time


SMALL_ENOUGH = 1e-3
GAMMA = 0.9
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
      # print ("  %s  |" % a)
      print( "| ",'{:3s}'.format(a),end=" ") 
    print( "| ")
  print ("-----------------------------")
  print ("")
  print ("")

if __name__ == '__main__':
  grid = standard_grid()
  print_values(grid.rewards, grid)
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

  print_policy(policy, grid)
  V = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      V[s] = np.random.random()
    else:
      V[s] = 0
  while True:
    biggest_change = 0
    for s in states:
      old_v = V[s]
      if s in policy:
        new_v = float('-inf')
        for a in ALL_POSSIBLE_ACTIONS:
          grid.set_state(s)
          r = grid.move(a)
          v = r + GAMMA * V[grid.current_state()]
          if v > new_v:
            new_v = v
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    if biggest_change < SMALL_ENOUGH:
      break

  for s in policy.keys():
    best_a = None
    best_value = float('-inf')
    for a in ALL_POSSIBLE_ACTIONS:
      grid.set_state(s)
      r = grid.move(a)
      v = r + GAMMA * V[grid.current_state()]
      if v > best_value:
        best_value = v
        best_a = a
    policy[s] = best_a

  print ("values:")
  print_values(V, grid)
  print ("policy:")
  print_policy(policy, grid)
