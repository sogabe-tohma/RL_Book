import numpy as np
import os, sys, time, datetime, json, random
from model import build_model
from buffer import Experience
from learn_maze import Qmaze
epsilon=0.1
def qtrain(model, maze, **opt):
    global epsilon
    n_epoch = opt.get('epochs',1001)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    
    qmaze = Qmaze(maze)
    
    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []   # history of win/lose game
    n_free_cells = len(qmaze.free_cells)
    hsize = qmaze.maze.size//2   # history window size
    win_rate = 0.0
    imctr = 1
   
    for epoch in range(n_epoch):
        states=[]
        loss = 0.0
        rat_cell = random.choice(qmaze.free_cells)
        if epoch%10==0:
            rat_cell = (2,3)
        qmaze.reset(rat_cell)
        game_over = False
        # get initial envstate (1d flattened canvas)
        envstate = qmaze.observe()
        n_episodes = 0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions:
                break
            prev_envstate = envstate
            states.append((qmaze.state[0],qmaze.state[1]))
            # Get next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            loss=model.fit(inputs, targets,1)
#             value= model.predict(inputs)
#             loss = rm.mean_squared_error(value, targets)
#             loss.grad().update(rm.Adam(0.001))

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} "
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate))  
        
    print(qmaze.draw(states))
if __name__ == "__main__":
    maze =  np.array([
    [ 1.,  1.,  1.,  1.],
    [ -1.0,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  1.]
    ])
    model = build_model()
    state=qtrain(model, maze, epochs=251, max_memory=8*maze.size, data_size=32)

