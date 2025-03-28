import random as r
import numpy as np
import matplotlib.pyplot as plt
import time as t

actions = ['up', 'down', 'left', 'right']

grid = [[-1, -1, -1, -1], # rewards for each unique state (s1...s12)
        [-1, -4, -1, -1], 
        [-1, -1, 6.4, 30]]

grid_enumerate = []
s, e = 1, len(grid[0]) + 1
for _ in range(len(grid)):
    grid_enumerate.append([*range(s, e)])
    s += 4
    e += 4

unique_states_len = len(grid) * len(grid[0])
Q_values_table = np.zeros((unique_states_len, len(actions)))
learning_rate, discount_rate = 0.1, 0.3



def Qlearning_update_stateAction(current_q, lr, reward, dr, next_state_row) -> float:
    return current_q + lr * (reward + dr * max(next_state_row) - current_q)



def choose_action(grid_p, epsilon=0.1) -> str:  
    assert grid_p[0] < len(grid) and grid_p[1] < len(grid[0]), 'Either x or y is out of bounds'
    
    if np.random.rand() < epsilon: 
        return r.choice(actions)

    unique_num = grid_enumerate[grid_p[0]][grid_p[1]]
    q_values = Q_values_table[unique_num - 1]
    
    actions_ = actions.copy()

    if grid_p[0] - 1 < 0:
        actions_.remove('up')
    if grid_p[0] + 1 >= len(grid):
        actions_.remove('down')
    if grid_p[1] - 1 < 0:
        actions_.remove('left')
    if grid_p[1] + 1 >= len(grid[0]):
        actions_.remove('right')

    if actions_:
        indices_left = [actions.index(act) for act in actions_]
        return actions_[np.argmax(q_values[indices_left])]



def take_one_action(grid_p, action):
    new_grid_p = grid_p.copy()
    if action == 'up' and grid_p[0] > 0:
        new_grid_p[0] -= 1
    elif action == 'down' and grid_p[0] < len(grid) - 1:
        new_grid_p[0] += 1
    elif action == 'left' and grid_p[1] > 0:
        new_grid_p[1] -= 1
    elif action == 'right' and grid_p[1] < len(grid[0]) - 1:
        new_grid_p[1] += 1
    return new_grid_p

print('Q_values table before optimization:\n', Q_values_table, '\n')

episodes = 10
paths = []
for episode in range(episodes):
    grid_point = [0, 0] 
    step_total_reward = 0
    end_of_grid = False
    path = []

    print(f"Episode {episode+1}")
    while not end_of_grid:
        if end_of_grid:
            print('--- end of grid, next episode ---\n')
        action = choose_action(grid_point)
        unique_num = grid_enumerate[grid_point[0]][grid_point[1]]
        action_idx = actions.index(action)

        next_grid_p = take_one_action(grid_point, action)
        match action:
            case 'up': print(f"{grid_point} /\ {next_grid_p}"); path.append('up')
            case 'down': print(f"{grid_point} \/ {next_grid_p}"); path.append('down')
            case 'left': print(f"{grid_point} <- {next_grid_p}"); path.append('left')
            case 'right': print(f"{grid_point} -> {next_grid_p}"); path.append('right')
                
        
        next_unique_num = grid_enumerate[next_grid_p[0]][next_grid_p[1]]
        reward = grid[next_grid_p[0]][next_grid_p[1]]

        Q_values_table[unique_num-1][action_idx] = Qlearning_update_stateAction(Q_values_table[unique_num-1][action_idx], learning_rate, reward, discount_rate, 
                                                                                Q_values_table[next_unique_num-1])
        grid_point = next_grid_p
        step_total_reward += reward

        end_of_grid = grid_point == [len(grid)-1, len(grid[0])-1]
    
    paths.append([path, step_total_reward])
    print(f"reward: {paths[episode][1]}\n")
    #t.sleep(0.5)

rewards = [r[1] for r in paths]
print('Q_values table after optimization:\n', Q_values_table, '\n')
plt.plot(range(episodes), rewards)
plt.show()

print('\n------------- Best Action for Each State (position) after Training -------------\n')
Q_values_table = Q_values_table.tolist()
max_indices = [Q_values_table[row_num].index(max(q_row)) for row_num, q_row in enumerate(Q_values_table)]
print([actions[idx] for idx in max_indices])

print('\n------------- Inference: Best Path from start to end -------------\n')
print(rewards)
print(paths[np.argmax(rewards)][0])

