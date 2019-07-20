
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
import time
import csv

np.random.seed(3585654)
GAMMA = 0.9
ALPHA=0.01
beta=100.0
ALL_POSSIBLE_ACTIONS =( 'L','R','U', 'D')
data={}
def random_action(a,eps=0.15):
    p = np.random.random()
    if p > (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

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

def play_game(grid,policy,for_total,forG):
    s = (3, 0)
    grid.set_state(s)
    a = random_action(policy[s])
    seen_states = set()
    steps=1
    t=count(a)
    for_total[s][t]=for_total[s][t]+1

    states_actions_rewards = [[s,a,0,for_total[s]]]
    while True:
        steps +=1
        old_s = grid.current_state()
        r = grid.move(a)
        s = grid.current_state()
        if grid.game_over():
            states_actions_rewards.append([s, None, r,[0,0,0,0]])
            break
        else:
            a = random_action(policy[s])
            t=count(a)
            for_total[s][t]=for_total[s][t]+1
            states_actions_rewards.append([s, a, r,for_total[s]])
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r,al in reversed(states_actions_rewards):
        
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G, al,forG[s]))
        G = r + GAMMA*G
    return states_actions_returns , steps
def convert_action(b):
    totl=[0,0,0,0]
    for t in ALL_POSSIBLE_ACTIONS:
        if b==t:
            c=ALL_POSSIBLE_ACTIONS .index(t)
            totl[c]=1
            break
    return totl
def count(b):
    c=0
    for t in ALL_POSSIBLE_ACTIONS:
        if b==t:
            c=ALL_POSSIBLE_ACTIONS .index(t)
            break
    return c
def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val
def softmax(datas):
    r=[v for k, v in datas.items()]
    e_x=np.exp(r-np.max(r))
    return (e_x/e_x.sum())

def soft_max(target,total):#
    sum1=0
    max1=0
    t=[v for k, v in total.items()]
    if max1 < np.max(t):
        max1=np.max(t)
    e_x=np.exp(beta*(target-max1))
    for te in range(4):
        sum1+=np.exp(beta*(t[te]-max1))
    final=(e_x/sum1)
    return final
if __name__ == '__main__':

    grid = standard_grid()
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    Pi = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions: 
            Pi[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Pi[s][a]=0.01
        else:
            pass
    deltas = []
    xdata = []
    ydata = []
    fig = plt.figure(figsize=(11,9),facecolor='white')
    axes = fig.gca()
    axes.set_xlim(0, 200)
    axes.set_ylim(-0.01, .02)
    line, = axes.plot(xdata, ydata, '-o' , linewidth = 1,markerfacecolor='white',\
            markeredgecolor="magenta", markersize = 1 )
    for T in range(500):
        for_total={}
        for_G={}
        states = grid.all_states()
        for s in states:
            for_total[s]=[0,0,0,0]
            for_G[s]=[0,0,0,0]
        step=0
        states_actions_returns,step= play_game(grid,policy,for_total,for_G)
        seen_state_action_pairs = set()
        biggest_change = 0
        for s, A, G ,al , G_al in states_actions_returns:
            old_Pis=Pi[s][A]
            sa=(s,A)
            g=[0,0,0,0]
            t=count(A)
            g[t]=G
            if sa not in seen_state_action_pairs:
                baseline=0
                delta=[]
                for a in ALL_POSSIBLE_ACTIONS:
                    t=0
                    t=count(a)
                    delta.append(al[count(a)]-np.sum(al)*soft_max(Pi[s][a],Pi[s])/step)
                for a in ALL_POSSIBLE_ACTIONS:
                    t=count(a)
                    Pi[s][a]=(Pi[s][a]+(ALPHA*delta[t])*(g[t]-baseline))
                diff=old_Pis-Pi[s][A]
                biggest_change = np.abs(diff)
                seen_state_action_pairs.add(sa)
        xdata.append(T)
        ydata.append(biggest_change)
        line.set_ydata(ydata)
        deltas.append(biggest_change)
        for s in policy.keys():
            policy[s] = max_dict(Pi[s])[0]
        value={}
        prob_p=[]
        for s, Qs in Pi.items():
            policy[s] = max_dict(Pi[s])[0]
            prob_pi=softmax(Pi[s])
            prob_p.append((s,prob_pi))
            value[s]= max_dict(Pi[s])[1]
        
        if T==1 or T==10 or T==100 or T==499:
            t={}
            n='Episod'
            tt=str(T)
            Name=n+tt
            for it in prob_p:
                t[str(it[0])]=np.round(it[1],2)
            data[Name]=t
  
    df=pd.DataFrame(data)
    df.to_csv("result.csv", index=True)
    print(df)
