from __future__ import print_function
import numpy as np
from grid_world import standard_grid
import matplotlib.pyplot as plt
import os
import sys
import time


SMALL_ENOUGH = 1e-2 
plt.figure(figsize=(9,7))

"""
orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f
"""


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
  for i in range(g.width):
    print ("--------------------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print ("  %s  |" % a,end="")
    print ("")

if __name__ == '__main__':
  grid = standard_grid()
  states = grid.all_states()
  V = {}
  for s in states:
    V[s] = 0
  gamma = 1 
  iterate = 0
  count =0

  labels   = ['(0,0)','(1,2)','(1,1)','(1,0)','(2,2)','(2,0)','(3,2)','(3,1)','(3,0)']
  while count < 5:
    biggest_change = 0
    count  = count +1
    for s in states:

      old_v = V[s]

      if s in grid.actions:
        iterate = iterate + 1
        new_v = 0 
        p_a = 1.0 / len(grid.actions[s]) 
        for a in grid.actions[s]:
          grid.set_state(s)

          r = grid.move(a)
          new_v += p_a * 1.0 * (r + gamma *  V[grid.current_state()])

        plt.scatter(iterate,V[(0,0)],c = "r",marker='o',s= 50,lw = 0,label = labels[0])
        labels[0] = "_nolegend_"
        plt.scatter(iterate,V[(1,2)],c = "g",marker='o',s= 50,lw = 0,label = labels[1])
        labels[1] = "_nolegend_"
        plt.scatter(iterate,V[(1,1)],c = "c",marker='o',s= 50,lw = 0,label = labels[2])
        labels[2] = "_nolegend_"
        plt.scatter(iterate,V[(1,0)],c = "black",s= 50,lw = 0,label = labels[3])
        labels[3] = "_nolegend_"
        plt.scatter(iterate,V[(2,2)],c = "y",s= 50,lw = 0,label = labels[4])
        labels[4] = "_nolegend_"
        plt.scatter(iterate,V[(2,0)],c = 'blue',s= 50,lw = 0,label = labels[5])
        labels[5] = "_nolegend_"
        plt.scatter(iterate,V[(3,2)],c = 'darkviolet',s= 50,lw = 0,label = labels[6])
        labels[6] = "_nolegend_"
        plt.scatter(iterate,V[(3,1)],c = 'cyan',s= 50,lw = 0,label = labels[7])
        labels[7] = "_nolegend_"
        plt.scatter(iterate,V[(3,0)],c = 'saddlebrown',s= 50,lw = 0,label = labels[8])
        labels[8] = "_nolegend_"
        plt.pause(0.0002)
        plt.xlim()
        plt.ylim(-100,70)
        V[s] = new_v
        plt.legend(loc = 'upper center',bbox_to_anchor=(0.5, 1.06),ncol=4, fancybox=True, shadow=True)

        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    print_values(V, grid)


    if biggest_change < SMALL_ENOUGH:
      break

plt.show()
