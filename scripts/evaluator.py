import torch
import numpy as np
from model import CNNModel, TransformerModel, TransformerMultiHeadModel
from env.env import MahjongGBEnv
from env.feature import FeatureAgent
class Evaluator:
    def __init__(self, config, model_ckpt=None, baseline_ckpt=None):
        self.device = config.get('device', 'cuda')
        self.eval_episodes = config.get('eval_episodes', 20)
        self.baseline_ckpt = config.get('baseline_ckpt', baseline_ckpt)

        self.env = MahjongGBEnv(config = {'agent_clz': FeatureAgent})

        self.config = config
        # 加载当前模型
        # self.model = CNNModel().to(self.device)
        # self.baseline_model = CNNModel().to(self.device)
        self.model = TransformerMultiHeadModel().to(self.device)
        self.baseline_model = TransformerMultiHeadModel().to(self.device)

        self.update_model(model_ckpt, baseline_ckpt)

    def update_model(self, model_ckpt=None, baseline_ckpt=None):
        if model_ckpt is not None:
            self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
            self.model.eval()
        if baseline_ckpt is not None:
            self.baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=self.device))
            self.baseline_model.eval()

    def evaluate(self):
        model_scores = []
        baseline_scores = []
        agent_names = ['player_%d' % i for i in range(1, 5)]
        baseline_agent = self.ModelAgent(0, self.baseline_model, self.device)
        eval_agent = self.ModelAgent(0, self.model, self.device)
        for seat in range(4):
            seat_name = f'player_{seat + 1}'
            for ep in range(self.eval_episodes // 4):

                obs = self.env.reset()

                done = False
                while not done:
                    action_dict = {}
                    for agent_name in obs:
                        state = obs[agent_name]
                        if agent_name == seat_name:
                            # use model to predict action
                            with torch.no_grad():
                                logits, value = eval_agent.act(state)
                                action_dist = torch.distributions.Categorical(logits = logits)
                                action = action_dist.sample().item()
                                value = value.item()
                        else:
                            with torch.no_grad():
                                logits, value = baseline_agent.act(state)
                                action_dist = torch.distributions.Categorical(logits = logits)
                                action = action_dist.sample().item()
                                value = value.item()
                        action_dict[agent_name] = action

                    # interact with env
                    obs, reward, done = self.env.step(action_dict)
                # 统计分数
                model_scores.append(reward[f'player_{seat+1}'])
                for i in range(4):
                    if i != seat:
                        baseline_scores.append(reward[f'player_{i+1}'])
        print(model_scores)
        avg_model = np.mean(model_scores)
        avg_baseline = np.mean(baseline_scores)
        
        return avg_model, avg_baseline

    class ModelAgent:
        def __init__(self, idx, model, device):
            self.idx = idx
            self.model = model
            self.device = device
        def act(self, obs):
            obs_tensor = torch.tensor(obs['observation']).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(obs['action_mask']).unsqueeze(0).to(self.device)

            return self.model({'observation': obs_tensor, 'action_mask': mask_tensor})[:2]