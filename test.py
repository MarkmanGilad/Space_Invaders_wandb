import numpy as np
import torch as T 

reward_arr = np.array([0.])
val_arr = np.array([20.59])
done_arr = np.array([1])



def calculate_advantage (reward_arr, val_arr, done_arr):
    gamma = 0.9
    gae_lambda = 0.95
    advantage = np.zeros(len(reward_arr), dtype=np.float32)

    future_advantage = 0
    for t in reversed(range(len(reward_arr))):
        if t == len(reward_arr) - 1:  
            td_error = reward_arr[t] - val_arr[t]  # No next value for last step
        else:
            td_error = reward_arr[t] + gamma * val_arr[t+1] * (1 - int(done_arr[t])) - val_arr[t]
            
        future_advantage = td_error + gamma * gae_lambda * future_advantage * (1 - int(done_arr[t]))
        advantage[t] = future_advantage
    
    
        advantage = T.tensor(advantage).to('cpu')
        # Normalization (optional)
        m = advantage.mean()
        s = advantage.std()
        
        advantage_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
    return advantage_norm

print(calculate_advantage(reward_arr, val_arr, done_arr))