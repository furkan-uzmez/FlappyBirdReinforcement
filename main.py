import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Agent():
    def __init__(self,hyperparameter_set):
        with open('hyperparameter.yml','r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            print(hyperparameters)


        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']


    def run(self,is_training=True , render=False):
        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states,num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

        reward_per_episode = []


        for episode in itertools.count():
            state , _ = env.reset()
            terminated = False
            episode_reward = 0.0
            while True:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()

                # Processing:
                new_state , reward, terminated, _, info = env.step(action)
                
                episode_reward += reward

                if is_training:
                    memory.push(state,action,reward,new_state,terminated)

                state = new_state

                # Checking if the player is still alive
                if terminated:
                    break
            
            reward_per_episode.append(episode_reward)
