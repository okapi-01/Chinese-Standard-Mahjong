import multiprocessing as mp
#mp.set_start_method('spawn', force=True)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.replay_buffer import ReplayBuffer
from scripts.actor_new import Actor
from scripts.learner import Learner
import time
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 25000, #50000
        'replay_buffer_episode': 300,
        'model_pool_size': 10, #20
        'model_pool_name': 'model-pool-transformer2',
        'num_actors': 48, #16
        'episodes_per_actor': 100000,
        'gamma': 0.99,
        'lambda': 0.95,
        'min_sample': 4096, #这俩可以再翻一倍
        'batch_size': 2048, #256,
        'epochs': 3,
        'clip': 0.2,
        'lr': 1e-6,
        'policy_coeff': 1.0,
        'value_coeff': 0.0001, #0.1
        'entropy_coeff': 0.003, #0.02,
        'kl_coeff': 0.0,
        'device': 'cuda',
        'ckpt_save_interval': 500,  # iter
        'ckpt_save_path': './models/',
        'pretrain_ckpt_path': './models/20260103-094027/72000.pt', #'ckpt/20251218-144714',
        #evaluate
        'eval_episodes': 100,
        'eval_interval': 500,  # iter
        'baseline_ckpt': 'ckpt/20251218-144714/20.pt',
    }
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config["ckpt_save_path"] = os.path.join(config["ckpt_save_path"], timestamp)
    os.makedirs(config["ckpt_save_path"], exist_ok=True)
    writer = SummaryWriter(log_dir=config["ckpt_save_path"])
    config['writer'] = writer
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    try:
        for actor in actors: actor.start()
        learner.start()
        for actor in actors: actor.join()
        learner.terminate()
    except KeyboardInterrupt:
        print("Exiting gracefully...")
        for actor in actors: actor.terminate()
        #learner.terminate()
    finally:
        writer.close()