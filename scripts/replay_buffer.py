from multiprocessing import Queue
from collections import deque
import numpy as np
import random

class ReplayBuffer:

    def __init__(self, capacity, episode):
        self.queue_win = Queue(episode)
        self.queue_lose = Queue(episode)
        self.capacity = capacity
        self.buffer = None
    
    def push_win(self, samples): # only called by actors
        self.queue_win.put(samples)
    def push_lose(self, samples): # only called by actors
        self.queue_lose.put(samples)
    
    def _flush(self):
        if self.buffer is None:
            self.buffer = deque(maxlen=self.capacity)
            self.stats = {'sample_in': 0, 'sample_out': 0, 'episode_in': 0}
        # 处理赢的队列
        while not self.queue_win.empty():
            episode_data = self.queue_win.get()
            unpacked_data = self._unpack(episode_data)
            for d in unpacked_data:
                d['win'] = True
            self.buffer.extend(unpacked_data)
            self.stats['sample_in'] += len(unpacked_data)
            self.stats['episode_in'] += 1
        # 处理输的队列
        while not self.queue_lose.empty():
            episode_data = self.queue_lose.get()
            unpacked_data = self._unpack(episode_data)
            for d in unpacked_data:
                d['win'] = False
            self.buffer.extend(unpacked_data)
            self.stats['sample_in'] += len(unpacked_data)
            self.stats['episode_in'] += 1
    
    def sample(self, batch_size, win_ratio=0.5):
        self._flush()
        assert len(self.buffer) > 0, "Empty buffer!"
        self.stats['sample_out'] += batch_size

        win_samples = [x for x in self.buffer if x.get('win', False)]
        lose_samples = [x for x in self.buffer if not x.get('win', False)]

        win_batch = int(batch_size * win_ratio)
        lose_batch = batch_size - win_batch

        batch = []
        if win_samples:
            batch += random.sample(win_samples, min(win_batch, len(win_samples)))
        if lose_samples:
            batch += random.sample(lose_samples, min(lose_batch, len(lose_samples)))

        # 补齐
        while len(batch) < batch_size and len(self.buffer) > len(batch):
            batch.append(random.choice(self.buffer))

        batch = self._pack(batch)
        return batch
    
    def size(self): # only called by learner
        self._flush()
        return len(self.buffer)
    
    def clear(self): # only called by learner
        self._flush()
        self.buffer.clear()
    
    def _unpack(self, data):
        # convert dict (of dict...) of list of (num/ndarray/list) to list of dict (of dict...)
        if type(data) == dict:
            res = []
            for key, value in data.items():
                values = self._unpack(value)
                if not res: res = [{} for i in range(len(values))]
                for i, v in enumerate(values):
                    res[i][key] = v
            return res
        else:
            return list(data)
            
    def _pack(self, data):
        # convert list of dict (of dict...) to dict (of dict...) of numpy array
        if type(data[0]) == dict:
            keys = data[0].keys()
            res = {}
            for key in keys:
                values = [x[key] for x in data]
                res[key] = self._pack(values)
            return res
        elif type(data[0]) == np.ndarray:
            return np.stack(data)
        else:
            return np.array(data)