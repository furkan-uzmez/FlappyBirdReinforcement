import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
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

 
    def run(self,is_training=True , render=False):
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
            memory = ReplayMemory(self.replay_memory_size)
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
            state = torch.tensor(state, dtype=torch.float32, device=device).flatten() # Convert state to tensor directly on device
            terminated = False
            episode_reward = 0.0
            while(not terminated and episode_reward < self.stop_on_reward):
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon :
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state , reward, terminated, _, info = env.step(action.item())
                
                episode_reward += reward

                new_state = torch.tensor(new_state,dtype=torch.float32, device = device).flatten()
                reward = torch.tensor(reward,dtype=torch.float, device = device)


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
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay , self.epsilon_min)
                    epsilon_history.append(epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
    
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
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

    def optimize(self,mini_batch,policy_dqn,target_dqn):
        states , actions , new_states , rewards , terminations = zip(*mini_batch)

         # Debug print
        for i, ns in enumerate(new_states[:5]):
            print(f"new_state[{i}] type: {type(ns)} shape: {ns.shape if isinstance(ns, torch.Tensor) else 'NA'}")


        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states , dim=1)

        print("states shape:", states.shape)
        print("new_states shape:", new_states.shape)


        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze() 
        
        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
     # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)