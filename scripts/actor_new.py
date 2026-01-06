import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multiprocessing import Process
import numpy as np
import torch

from scripts.replay_buffer import ReplayBuffer
from scripts.model_pool import ModelPoolClient
from env.env import MahjongGBEnv
from env.feature import FeatureAgent
from model import CNNModel, TransformerModel, TransformerMultiHeadModel

class Actor(Process):

    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        
    def run(self):
        torch.set_num_threads(1)
        device = torch.device('cpu')
    
        #self.pretrained_model = CNNModel() # pretrained model
        self.pretrained_model = TransformerMultiHeadModel() # pretrained model
        if self.config['baseline_ckpt']:
                self.pretrained_model.load_state_dict(torch.load(self.config['baseline_ckpt'], map_location=device))
        else:
                raise FileNotFoundError("No pre-trained model found in the specified path.")
        self.pretrained_model.to(device) 
        self.pretrained_model.eval() # 确保是 eval 模式

        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        
        # print("actor running")
        # create network model
        #model = CNNModel()
        model = TransformerMultiHeadModel()
        
        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        model.load_state_dict(state_dict)
        model.to(device)
        
        # collect data
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})
        # policies = {player : model for player in env.agent_names} # all four players use the latest model
        rotation_counter = 0
        for episode in range(self.config['episodes_per_actor']):
            # print(episode)
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                model.load_state_dict(state_dict)
                version = latest
            
            # rotate player position
            current_position = rotation_counter % 4
            rotation_counter += 1
            
            # make player policies currend model + 3 pretrained models
            policies = {}
            for i, agent_name in enumerate(env.agent_names):
                if i == current_position:
                    policies[agent_name] = model
                else:
                    policies[agent_name] = self.pretrained_model
            
            # only collect data for the current model
            train_agent_name = env.agent_names[current_position]
            
            # run one episode and collect data
            obs = env.reset()
            episode_data = {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [],
                'value' : [],
                'log_prob' : [],
            }
            done = False
            ppp = 1
            while not done:
                # print(ppp)
                ppp += 1
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    # agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    
                    if agent_name == train_agent_name:
                        episode_data['state']['observation'].append(state['observation'])
                        episode_data['state']['action_mask'].append(state['action_mask'])
                    state['observation'] = torch.tensor(state['observation'], dtype = torch.float).unsqueeze(0).to(device)
                    state['action_mask'] = torch.tensor(state['action_mask'], dtype = torch.float).unsqueeze(0).to(device)
                    # model.train(False) # Batch Norm inference mode
                    with torch.no_grad():
                        current_model = policies[agent_name]
                        current_model.eval()
                        logits, value = current_model(state)
                        logits = logits.cpu()
                        value = value.cpu()
                        if agent_name == train_agent_name:
                            action_dist = torch.distributions.Categorical(logits = logits)
                            action = action_dist.sample().item()
                            
                            log_prob = action_dist.log_prob(torch.tensor(action)).item()
                            episode_data['log_prob'].append(log_prob)
                            
                            value = value.item()
                            episode_data['action'].append(action)
                            episode_data['value'].append(value)
                        else:
                            action = torch.argmax(logits, dim=1).item()
                        actions[agent_name] = action
                        
                # interact with env
                next_obs, rewards, done = env.step(actions)
                
                # record rewards
                #print(rewards)
                if train_agent_name in rewards:
                    rew = rewards[train_agent_name]
                    
                    if rew > 20:
                        rew = 20 + 0.1*(rew-20)
                    if rew < -20:
                        rew = -20 + 0.1*(rew+20)
                    episode_data['reward'].append(rew)
                    # episode_data['reward'].append(rewards[train_agent_name])
                # else:
                #     episode_data['reward'].append(0)
                obs = next_obs
            #print(self.name, 'Episode', episode, 'Model', latest['id'], 'Reward', rewards)
            
            # postprocessing episode data
            #print(episode_data['reward'])
            if len(episode_data['state']['observation']) == 0:
                continue
            if len(episode_data['action']) < len(episode_data['reward']):
                episode_data['reward'].pop(0)
            obs = np.stack(episode_data['state']['observation'])
            mask = np.stack(episode_data['state']['action_mask'])
            actions = np.array(episode_data['action'], dtype = np.int64)
            rewards = np.array(episode_data['reward'], dtype = np.float32)
            values = np.array(episode_data['value'], dtype = np.float32)
            log_probs = np.array(episode_data['log_prob'], dtype = np.float32)
            next_values = np.array(episode_data['value'][1:] + [0], dtype = np.float32)
            
            # print(actions.shape, rewards.shape, values.shape, next_values.shape)
            td_target = rewards + next_values * self.config['gamma']
            td_delta = td_target - values
            advs = []
            adv = 0
            for delta in td_delta[::-1]:
                adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                advs.append(adv) # GAE
            advs.reverse()
            advantages = np.array(advs, dtype = np.float32)
            lambda_returns = values + advantages
            if rewards[-1] > 0:
                self.replay_buffer.push_win({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': lambda_returns, #'target': td_target,
                    'reward': rewards,
                    'log_prob': log_probs,
                })
            else:
                self.replay_buffer.push_lose({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': lambda_returns, #'target': td_target,
                    'reward': rewards,
                    'log_prob': log_probs,
                })
