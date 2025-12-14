from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F

from scripts.replay_buffer import ReplayBuffer
from scripts.model_pool import ModelPoolServer
from model import CNNModel
import os
from torch.utils.tensorboard import SummaryWriter
from evaluator import Evaluator
import wandb
class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        #self.evaluator = Evaluator(self.config)
        self.evaluator = None
        
    
    def run(self):
        wandb.init(
            project="mahjong-rl",
            group="experiment",  # 多进程用同一个 group
            name=f"learner_{os.getpid()}",
            dir=self.config["ckpt_save_path"],
        )
        writer = self.config['writer']
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        iterations = 0
        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNModel()
        if self.config['pretrain_ckpt_path']:
            f = self.config['pretrain_ckpt_path']
            model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
            print(f"Loaded pre-trained model from {f}")
        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
 
        while True:
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            states = {
                'observation': obs,
                'action_mask': mask
            }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            targets = torch.tensor(batch['target']).to(device)
            
            
            
            # calculate PPO loss
            model.train(True) # Batch Norm training mode
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
            old_log_probs = torch.log(old_probs + 1e-8).detach()
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                action_dist = torch.distributions.Categorical(logits = logits)
                probs = F.softmax(logits, dim = 1).gather(1, actions)
                log_probs = torch.log(probs + 1e-8)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()

            # writer.add_scalar('Loss/total', total_loss / self.config['epochs'], iterations)
            # writer.add_scalar('Loss/policy', total_policy_loss / self.config['epochs'], iterations)
            # writer.add_scalar('Loss/value', total_value_loss / self.config['epochs'], iterations)
            # writer.add_scalar('Loss/entropy', total_entropy_loss / self.config['epochs'], iterations)
            wandb.log({
                "Loss/total": total_loss / self.config['epochs'],
                "Loss/policy": total_policy_loss / self.config['epochs'],
                "Loss/value": total_value_loss / self.config['epochs'],
                "Loss/entropy": total_entropy_loss / self.config['epochs'],
            })
            #print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))
            print('[Train] Iteration %d, total loss %.4f, policy loss %.4f, value loss %.4f, entropy loss %.4f' % (
                iterations, total_loss / self.config['epochs'], total_policy_loss / self.config['epochs'],
                total_value_loss / self.config['epochs'], total_entropy_loss / self.config['epochs']))
            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)
            
            # save checkpoints
            if iterations % self.config["ckpt_save_interval"] == 0:
                path = os.path.join(self.config['ckpt_save_path'], '%d.pt' % iterations)
                torch.save(model.state_dict(), path)

            if iterations % self.config["eval_interval"] == 0 and self.config.get('baseline_ckpt', None) is not None:
                if self.evaluator is None:
                    self.evaluator = Evaluator(self.config, None, self.config['baseline_ckpt'])

                self.evaluator.update_model(model_ckpt=os.path.join(self.config['ckpt_save_path'], '%d.pt' % iterations),)
                avg_model, avg_baseline = self.evaluator.evaluate()
                print(f'[Evaluate] Iteration {iterations}: Model avg: {avg_model:.2f} | Baseline avg: {avg_baseline:.2f}')
                # writer.add_scalar('eval/model_avg', avg_model, iterations)
                # writer.add_scalar('eval/baseline_avg', avg_baseline, iterations)
                wandb.log({
                    "eval/reward_model": avg_model,
                    "eval/reward_baseline": avg_baseline,
                })
            iterations += 1