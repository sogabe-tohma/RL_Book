"""
Dependencies:
tensorflow r1.2
keras 2.2.4
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
from model import Model
from buffer import Experience
from learn_grid import Qgrid
epsilon=0.1

def qtrain(model, grid, **opt):
    global epsilon
    n_epoch = opt.get('epochs',1001)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    actions=['Left','Up','Right','Down']
    qgrid = Qgrid(grid)
    experience = Experience(model, max_memory=max_memory)

    n_free_cells = len(qgrid.free_cells)
    hsize = qgrid.grid.size//2
    win_rate = 0.0
    imctr = 1
    all_state=[]
    for epoch in range(n_epoch):
        states=[]
        all_action=[]
        loss = 0.0
        rat_cell = random.choice(qgrid.free_cells)
        rat_cell = (2,3)
        qgrid.reset(rat_cell)
        game_over = False
        envstate = qgrid.observe()
        n_episodes = 0
        while not game_over:
            valid_actions = qgrid.valid_actions()
            if not valid_actions:
                break
            prev_envstate = envstate
            states.append((qgrid.state[0],qgrid.state[1]))
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))
            all_action.append(actions[action])

            envstate, reward, game_status = qgrid.act(action)
            if game_status == 'win':
                game_over = True
            elif game_status == 'lose':
                game_over = True
            else:
                game_over = False
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1
            inputs, targets = experience.get_data(data_size=data_size)
            model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)
        print("epoch",epoch, 'actions',all_action)
        all_state.append((states))
    qgrid.draw(all_state)


if __name__ == "__main__":
    grid =  np.array([
    [ 1.,  1.,  1.,  1.],
    [ -1.0,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  1.]
    ])
    _model = Model()
    model=_model.build_model(grid)
    qtrain(model, grid, epochs=51, max_memory=8*grid.size, data_size=32)
