import tensorflow as tf
import numpy as np
import ray
import time

def ray_get_and_free(object_ids):
    
def mlp(x, hidden_size, activation, out_activation=None):
    for hidden in hidden_size[:-1]:
        x = tf.layers.dense(x, hidden, activation=activation)
    return tf.layers.dense(x, hidden_size[-1], activation=out_activation)

class Model:
    '''
    Model for evaluation, used directly in workers
    when used in learner, need to add loss and train_ops
    '''
    def __init__(self, obs_dim, act_dim, act_lim, **kwargs):
        self.obs = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim))
        self.act = tf.placeholder(dtype=tf.float32, shape=(None, act_dim))
        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            self.pi = act_lim * mlp(
                self.obs, [256, 256, act_dim], tf.nn.relu, tf.nn.tanh
            )
        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            self.q = mlp(
                tf.concat([self.obs, self.act]),
                [256, 256, 1], tf.nn.relu, None
            )
        with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
            self.q_pi = mlp(
                tf.concat([self.obs, self.pi]),
                [256, 256, 1], tf.nn.relu, None
            )
    
    def load_from_weights(self, weights):
        pass

    def get_weights(self):
        pass

@ray.remote
class AsyncParameterServer():
    '''
    Asyncronous parameter server, 
    used for parameter sharing between learner and workers
    '''
    def __init__(self):
        self._param = None
        # hashing is a notation for paramter version
        self.hashing = int(time.time())
    
    def pull(self):
        return self._param
    
    def push(self, new_param):
        self._param = new_param
        self.hashing = int(time.time())
    
    def get_hashing(self):
        return self.hashing

def build_worker_model(kwargs):
    return Model(act_dim=kwargs['act_dim'], 
                 obs_dim=kwargs['obs_dim'],
                 act_lim=kwargs['act_lim'])

def build_worker_env(kwargs):
    import gym
    return gym.make(kwargs['env_name'])

@ray.remote
class Worker():
    def __init__(self, model_fn, env_fn, kwargs):
        self.id = int(kwargs['id'])
        self.noise_scale = kwargs['noise_scale']
        self.t_max = kwargs['t_max']
        
        self.env = env_fn(env_args)
        model_args['act_space'] = self.env.action_space
        model_args['obs_space'] = self.env.observation_space
        self.model = model_fn(model_args)

        self.ckpt = None
        self.ckpt_dir = kwargs['ckpt_dir']
        self.load_ckpt_period = int(kwargs['load_ckpt_period'])

        # ps means parameter server
        self.ps = kwargs['ps']
        self.sess = tf.Session()
        self._init_param()
    
    def _init_param(self):
        ckpt_s = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt_s and ckpt_s.model_checkpoint_path:
            self.saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_n_hours=1)
            self.saver.restore(self.sess, ckpt_s.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        print("init worker {} checkpoint hashing".format(self.id))
        self.ckpt_hashing = ray.get(self.ps.get_hashing.remote())
    
    def _data_generator(self):
        ep_step, ep_score, d, obs = 0, 0, False, self.env.reset()
        while True:
            # load checkpoint from parameter server if necessary
            ckpt_hashing_ = ray.get(self.ps.get_hashing.remote())
            if ckpt_hashing_ != self.ckpt_hashing:
                weights = ray.get(self.ps.pull.remote())
                if weights:
                    self.model.load_from_weights(weights)
                    print("worker {} load weights from parameter server, before:{}, after:{}".format(
                        self.id, self.ckpt_hashing, ckpt_hashing_))
                    self.ckpt_hashing = ckpt_hashing_
                else:
                    print("worker {} load None from parameter server!".format(self.id))
            
            # step in environment
            assert not d and ep_step < self.t_max
            act = self.sess.run(self.model.pi, {self.model.obs: obs})
            act += self.noise_scale * self.model.act_lim
            act = np.clip(act, -self.model.act_lim, self.model.act_lim)

            obs_, r, d, _ = self.env.step(act)
            ep_step += 1
            ep_score += 1
            yield (obs, act, obs_, r, d)

            if d or ep_step >= self.t_max:
                ep_step, ep_score, d, obs = 0, 0, False, self.env.reset()
    
    def get(self):
        return next(self._data_generator())

class RolloutCollector():
    def __init__(self, num_workers, ps, **kwargs):
        self.num_workers = int(num_workers)
        self.ps = ps
        self.workers = [
            Worker(
                model_fn=build_worker_model,
                env_fn=build_worker_env,
                kwargs=kwargs
            ).remote(
                num_cpus=kwargs['cpu_per_actor']
            ) for _ in range(num_workers)
        ]
        print("Workers starting ......")

    def _data_generator(self, num_returns=1, timeout=None):
        self.worker_done = [True for _ in range(self.num_workers)]
        self.working_job_ids = []
        self.id2job_idx = dict()
        while True:
            for i in range(self.num_workers):
                if self.worker_done[i]:
                    job_id = self.workers[i].get.remote()
                    self.working_job_ids.append(job_id)
                    self.id2job_idx[job_id] = i
                    self.worker_done[i] = False

            ready_ids, self.working_ids = ray.wait(
                self.working_job_ids, num_returns, timeout
            )

            for ready_id in ready_ids:
                self.worker_done[self.id2job_idx[ready_id]] = True
                self.id2job_idx.pop(ready_id)
            
            yield ready_ids






        
