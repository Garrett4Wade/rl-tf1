'''
Conclusion:
MADDPG ReplayBuffer can be generalized without passing obs_dim and act_dim, 
and doesn't even need to pad sequence, but is ~30 times slower than numpy array ones.
'''
import numpy as np
import random
import time

class OffpolicyReplayBuffer():
    def __init__(self, obs_dim, act_dim, max_size):
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.nex_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.float32)
        self.r = np.zeros(max_size, dtype=np.float32)
        self.d = np.zeros(max_size, dtype=np.bool)
        self.iter = 0
        self.max_size = max_size
    
    def add(self, obs, act, nex_obs, r, d):
        idx = self.iter % self.max_size
        self.obs[idx] = obs
        self.act[idx] = act
        self.nex_obs[idx] = nex_obs
        self.r[idx] = r
        self.d[idx] = d
        self.iter += 1
    
    def get(self, batch_size):
        batch_idx = np.random.choice(min(self.iter, self.max_size), batch_size)
        return dict(obs=self.obs[batch_idx],
                    act=self.act[batch_idx],
                    nex_obs=self.nex_obs[batch_idx],
                    r=self.r[batch_idx],
                    d=self.d[batch_idx])

class ReplayBuffer():
    def __init__(self, size):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


if __name__ == "__main__":
    obs_dim = 8
    act_dim = 4

    buffer1 = OffpolicyReplayBuffer(obs_dim, act_dim, int(1e6))
    buffer2 = ReplayBuffer(int(1e6))
    for _ in range(int(1e6)):
        obs, a, nex_obs, r, d = \
            np.random.randn(obs_dim), \
            np.random.randn(act_dim), \
            np.random.randn(obs_dim), \
            np.random.randn(), bool(np.random.randn())
        buffer1.add(obs, a, nex_obs, r, d)
        buffer2.add(obs, a, r, nex_obs, d)
    
    start = time.time()
    batch1 = buffer1.get(32000)
    end = time.time()
    dur1 = end - start

    start = time.time()
    batch2 = buffer2.sample(32000)
    end = time.time()
    dur2 = end - start

    print("Type 1 buffer cost time {}".format(dur1))
    print("Type 2 buffer cost time {}".format(dur2))
    
