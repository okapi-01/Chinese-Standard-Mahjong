from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F

from scripts.replay_buffer import ReplayBuffer
from scripts.model_pool import ModelPoolServer
from model import CNNModel, TransformerModel, TransformerMultiHeadModel
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
        self.best_model = -float('inf')
        
    
    def run(self):
        # wandb.init(
        #     project="mahjong-rl",
        #     group="experiment",  # 多进程用同一个 group
        #     name=f"learner_{os.getpid()}",
        #     dir=self.config["ckpt_save_path"],
        # )
        #writer = self.config['writer']
        writer = SummaryWriter(log_dir=self.config["ckpt_save_path"])
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        iterations = 0
        # initialize model params
        device = torch.device(self.config['device'])
        #model = CNNModel()
        model = TransformerMultiHeadModel(dropout=0)
        if self.config['pretrain_ckpt_path']:
            model_files = [f for f in os.listdir(self.config['pretrain_ckpt_path']) if f.endswith('.pt')]
            if model_files:
                max_epoch = max([int(f.split('.')[0]) for f in model_files if f.split('.')[0].isdigit()])
                model_path = os.path.join(self.config['pretrain_ckpt_path'], f"{max_epoch}.pt")
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                iterations = max_epoch + 1
                print(f"Loaded pre-trained model from {model_path}")
            else:
                raise FileNotFoundError("No pre-trained model found in the specified path.")
        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])


        while True:
            # 1. 等待足够的数据 (Rollout)
            while self.replay_buffer.size() < self.config['min_sample']:
                time.sleep(1)
            
            # 2. 一次性取出所有数据 (或者 min_sample 大小的数据)
            # 建议直接取 min_sample 大小，作为一次 PPO 更新的完整数据集
            rollout_size = self.config['min_sample'] 
            full_batch = self.replay_buffer.sample(rollout_size)
            
            # 3. 【关键】清空 Buffer，确保下次训练用的都是新策略生成的数据
            self.replay_buffer.clear()
            
            # 4. 准备全量数据 Tensor
            obs = torch.tensor(full_batch['state']['observation']).to(device)
            mask = torch.tensor(full_batch['state']['action_mask']).to(device)
            actions = torch.tensor(full_batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(full_batch['adv']).to(device)
            targets = torch.tensor(full_batch['target']).to(device)
            old_log_probs = torch.tensor(full_batch['log_prob']).unsqueeze(-1).to(device)
            
            # 归一化 Advantage (对整个 Batch 进行)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            advs = advs.unsqueeze(-1)

            # 5. Mini-batch 训练
            # PPO 标准做法：在收集到的大 Batch 上，切分成小 Mini-batch 训练多轮
            dataset_size = len(actions)
            indices = np.arange(dataset_size)
            mini_batch_size = self.config['batch_size'] # 使用 config 中的 batch_size 作为 mini-batch 大小
            
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            total_kl = 0
            update_count = 0

            for _ in range(self.config['epochs']):
                np.random.shuffle(indices) # 每个 Epoch 打乱顺序
                
                for start in range(0, dataset_size, mini_batch_size):
                    end = start + mini_batch_size
                    idx = indices[start:end]
                    
                    # 切片获取 Mini-batch
                    mb_obs = obs[idx]
                    mb_mask = mask[idx]
                    mb_actions = actions[idx]
                    mb_advs = advs[idx]
                    mb_targets = targets[idx]
                    mb_old_log_probs = old_log_probs[idx]
                    
                    mb_states = {
                        'observation': mb_obs,
                        'action_mask': mb_mask
                    }
                    
                    # --- 以下是原本的训练逻辑，注意变量名替换为 mb_ 前缀 ---
                    logits, values = model(mb_states)
                    action_dist = torch.distributions.Categorical(logits = logits)
                    # probs = F.softmax(logits, dim = 1).gather(1, mb_actions) # 没用到，注释掉
                    log_probs = F.log_softmax(logits, dim=1).gather(1, mb_actions)
                    
                    # 计算 KL (用于监控)
                    with torch.no_grad():
                        approx_kl = (mb_old_log_probs - log_probs).mean().item()
                    
                    # PPO Loss 计算
                    ratio = torch.exp(log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * mb_advs
                    
                    # 加权 Policy Loss (保持你之前的逻辑)
                    mask_play = ((mb_actions >= 2) & (mb_actions < 36)).float()
                    mask_int = 1.0 - mask_play
                    loss_elementwise = -torch.min(surr1, surr2)
                    sample_weights = mask_play * 0.4 + mask_int * 0.6
                    policy_loss = (loss_elementwise * sample_weights).mean()
                    
                    value_loss = torch.mean(F.mse_loss(values.squeeze(-1), mb_targets))
                    entropy_loss = -torch.mean(action_dist.entropy())
                    kl_loss = torch.mean(mb_old_log_probs - log_probs) # 如果 kl_coeff 设为 0，这项就没用了
                    
                    loss = self.config['policy_coeff'] * policy_loss + \
                            self.config['value_coeff'] * value_loss + \
                            self.config['entropy_coeff'] * entropy_loss + \
                            self.config['kl_coeff'] * kl_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    # 累加统计
                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    total_kl += approx_kl
                    update_count += 1

            # 6. 记录日志 (取平均值)
            avg_loss = total_loss / update_count
            avg_policy_loss = total_policy_loss / update_count
            avg_value_loss = total_value_loss / update_count
            avg_entropy_loss = total_entropy_loss / update_count
            avg_kl = total_kl / update_count
            
            iterations += 1 # 这里的 iterations 定义为完成一轮“收集+训练”
            
            writer.add_scalar('Loss/total', avg_loss, iterations)
            writer.add_scalar('Loss/policy', avg_policy_loss, iterations)
            writer.add_scalar('Loss/value', avg_value_loss, iterations)
            writer.add_scalar('Loss/entropy', avg_entropy_loss, iterations)
            writer.add_scalar('KL/divergence', avg_kl, iterations)
            
            print('[Train] Iteration %d, total loss %.4f, policy loss %.4f, value loss %.4f, entropy loss %.4f, KL: %.4f' % (
                iterations, avg_loss, avg_policy_loss, avg_value_loss, avg_entropy_loss, avg_kl))
            

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
                writer.add_scalar('eval/model_avg', avg_model, iterations)
                writer.add_scalar('eval/baseline_avg', avg_baseline, iterations)
                
                rewards = full_batch['reward']
                mean_reward = np.mean(rewards)
                writer.add_scalar('train/mean_reward', mean_reward, iterations)

                if avg_model > self.best_model:
                    self.best_model = avg_model
                    path = os.path.join(self.config['ckpt_save_path'], 'best.pt')
                    torch.save(model.state_dict(), path)
                # wandb.log({
                #     "eval/reward_model": avg_model,
                #     "eval/reward_baseline": avg_baseline,
                # })
            iterations += 1