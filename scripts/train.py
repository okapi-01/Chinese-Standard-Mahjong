import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
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
        'replay_buffer_size': 100000,
        'replay_buffer_episode': 400,
        'model_pool_size': 20,
        'model_pool_name': 'model-pool',
        'num_actors': 32,
        'episodes_per_actor': 100000,
        'gamma': 0.99,
        'lambda': 0.95,
        'min_sample': 1024,
        'batch_size': 256,
        'epochs': 1,
        'clip': 0.2,
        'lr': 5e-5,
        'value_coeff': 1,
        'entropy_coeff': 0.02,
        'device': 'cuda',
        'ckpt_save_interval': 3000,  # iter
        'ckpt_save_path': './models/',
        'pretrain_ckpt_path': 'ckpt/20251230-175913/100.pkl',
        #evaluate
        'eval_episodes': 500,
        'eval_interval': 3000,  # iter
        'baseline_ckpt': 'ckpt/20251213-160350/100.pkl',
        'model': 'CNN2',
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