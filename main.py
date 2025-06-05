import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import PrioritizedReplayMemory
import itertools
import yaml
import random
from torch import nn
import os
import matplotlib
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR,exist_ok=True)

matplotlib.use('Agg')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = "cpu"

class Agent():
    def __init__(self,hyperparameter_set):
        with open('hyperparameters.yml','r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']

        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params',{})

        self.LOG_FILE = os.path.join(RUNS_DIR,f"{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR,f"{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR,f"{self.hyperparameter_set}.png")

        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']     # dueling dqn on/off flag


        self.per_beta = hyperparameters.get('per_beta', 0.4)
        self.per_alpha = hyperparameters.get('per_alpha', 0.6)

 
    def run(self,is_training=True , render=True):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
        env = gymnasium.make(self.env_id, render_mode="human" if render else None,**self.env_make_params)

        num_states = int(np.prod(env.observation_space.shape))  # flatten size
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states,num_actions,self.fc1_nodes).to(device)

        reward_per_episode = []


        if is_training:
            memory = PrioritizedReplayMemory(self.replay_memory_size, alpha=self.per_alpha)
            epsilon = self.epsilon_init
            
            
            target_dqn = DQN(num_states,num_actions,self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters() , lr=self.learning_rate)

            epsilon_history = []

            best_reward = -9999999
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()

        for episode in itertools.count():
            state , _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device
            terminated = False
            episode_reward = 0.0
            while(not terminated and episode_reward < self.stop_on_reward):
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state , reward, terminated, _, info = env.step(action.item())
                
                episode_reward += reward

                # FIX: Convert to tensor with proper shape
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                if is_training:
                    memory.push(state,action,reward,new_state,terminated)

                    step_count += 1

                state = new_state
            
            reward_per_episode.append(episode_reward)

            
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                if len(memory) > self.mini_batch_size:
                    mini_batch, indices, weights = memory.sample(self.mini_batch_size, beta=self.per_beta)
                    self.optimize(mini_batch, weights, indices, memory, policy_dqn, target_dqn)


                    epsilon = max(epsilon * self.epsilon_decay , self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
    
    def save_graph(self, reward_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(reward_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(reward_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, weights, indices, memory, policy_dqn, target_dqn):
        states, actions, rewards, new_states, terminations = zip(*mini_batch)

        # Debug: Check shapes before stacking
        """
        print("Debug - tensor shapes before stacking:")
        print(f"  states[0]: shape={states[0].shape}, type={type(states[0])}")
        print(f"  new_states[0]: shape={new_states[0].shape}, type={type(new_states[0])}")
        print(f"  actions[0]: shape={actions[0].shape}, type={type(actions[0])}")
        print(f"  rewards[0]: shape={rewards[0].shape}, type={type(rewards[0])}")
        """
        # Stack tensors properly
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)  
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)
        
        #print(f"After stacking - new_states shape: {new_states.shape}")

        with torch.no_grad():
             if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
             else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze() 

        td_errors = (current_q - target_q).detach().abs() + 1e-5

        # Update priorities
        memory.update_priorities(indices, td_errors.cpu().numpy())
        
        loss = (self.loss_fn(current_q, target_q) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    print('Device : ',device)
     # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)
    
    
    # python main.py cartpole1 --train



    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)