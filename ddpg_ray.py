import tensorflow as tf
import numpy as np
import ray
import time
from threading import Thread
from replay_buffer import OffpolicyReplayBuffer
import re

_LAST_FREE_TIME = 0.0
_TO_FREE = []

def ray_get_and_free(object_ids):
    """
    Call ray.get and then queue the object ids for deletion.

    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.

    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.

    Returns:
        The result of ray.get(object_ids).
    """

    free_delay_s = 10.0
    max_free_queue_size = 100

    global _LAST_FREE_TIME
    global _TO_FREE

    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _TO_FREE.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_TO_FREE) > max_free_queue_size
            or now - _LAST_FREE_TIME > free_delay_s):
        ray.internal.free(_TO_FREE)
        _TO_FREE = []
        _LAST_FREE_TIME = now

    return result

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def mlp(x, hidden_size, activation, out_activation=None):
    for hidden in hidden_size[:-1]:
        x = tf.layers.dense(x, hidden, activation=activation)
    return tf.layers.dense(x, hidden_size[-1], activation=out_activation)

def ddpg_mlp_actor_critic(obs, 
                          a, 
                          act_dim,
                          act_lim,
                          hidden_size=(256, 256), 
                          activation=tf.nn.relu,
                          out_activation=tf.nn.tanh,
                          ):
    with tf.variable_scope('pi'):
        pi = act_lim * mlp(obs, list(hidden_size)+[act_dim], activation, out_activation)
    with tf.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([obs, a], -1), list(hidden_size)+[1], activation), -1)
    with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
        q_pi = tf.squeeze(mlp(tf.concat([obs, pi], -1), list(hidden_size)+[1], activation), -1)
    return pi, q, q_pi

class Model:
    '''
    Model for evaluation, used directly in workers
    when used in learner, need to add loss and train_ops
    '''
    def __init__(self, obs_dim, act_dim, act_lim, is_training):
        self.set_ws_ops = None
        self.ws_dphs = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_lim = act_lim
        self.obs, self.act, self.nex_obs, self.r, self.d = \
            tf.placeholder(dtype=tf.float32, shape=(None, obs_dim), name='obs'), \
            tf.placeholder(dtype=tf.float32, shape=(None, act_dim), name='act'), \
            tf.placeholder(dtype=tf.float32, shape=(None, obs_dim), name='nex_obs'), \
            tf.placeholder(dtype=tf.float32, shape=(None,), name='r'), \
            tf.placeholder(dtype=tf.float32, shape=(None,), name='d')
        with tf.variable_scope('main'):
            self.pi, self.q, self.q_pi = ddpg_mlp_actor_critic(self.obs, 
                                                               self.act,
                                                               act_dim, act_dim)
        if is_training:
            with tf.variable_scope('target'):
                _, _, self.q_pi_tgt = ddpg_mlp_actor_critic(self.obs, 
                                                                   self.act,
                                                                   act_dim, act_dim)
            self.target_init_op = tf.group([tf.assign(tgt_var, var)
                                    for var, tgt_var in zip(get_vars("main"), get_vars("target"))])

    def load_from_weights(self, sess, ws):
        if self.set_ws_ops is None:
            self.set_ws_ops = self._set_ws(ws)
        fd = dict()
        for k, v in self.ws_dphs.items():
            assert k in ws
            fd[v] = ws[k]
        sess.run(self.set_ws_ops, feed_dict=fd)

    def get_weights(self, sess):
        tvars = tf.trainable_variables()
        ws = sess.run(tvars)
        names = [re.match(
            "^(.*):\\d+$", var.name).group(1) for var in tvars]
        return dict(zip(names, ws))

    def _set_ws(self, to_ws):
        """

        :param to_ws name2var
        :return: run the ops
        """
        tvars = tf.trainable_variables()
        print('tvars', tvars)
        names = [re.match(
            "^(.*):\\d+$", var.name).group(1) for var in tvars]
        ops = []
        names_to_tvars = dict(zip(names, tvars))
        dphs = dict()
        for name, var in names_to_tvars.items():
            assert name in to_ws
            ph = tf.placeholder(dtype=to_ws[name].dtype, shape=to_ws[name].shape)
            dphs[name] = ph
            op = tf.assign(var, ph)
            ops.append(op)
        self.ws_dphs = dphs
        return tf.group(ops)

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

def build_model(kwargs):
    return Model(act_dim=kwargs['act_dim'], 
                 obs_dim=kwargs['obs_dim'],
                 act_lim=kwargs['act_lim'],
                 is_training=False)

def build_env(kwargs):
    import gym
    return gym.make(kwargs['env_name'])

def build_training_model(kwargs):
    return Model(act_dim=kwargs['act_dim'], 
                 obs_dim=kwargs['obs_dim'],
                 act_lim=kwargs['act_lim'],
                 is_training=True)

class Worker():
    def __init__(self, worker_id, model_fn, env_fn, ps, kwargs):
        self.id = worker_id
        self.noise_scale = kwargs['noise_scale']
        self.t_max = kwargs['t_max']
        
        self.env = env_fn(kwargs)
        self.model = model_fn(kwargs)

        self.ckpt = None
        self.ckpt_dir = kwargs['ckpt_dir']
        self.load_ckpt_period = int(kwargs['load_ckpt_period'])

        # ps means parameter server
        self.ps = ps
        self.sess = tf.Session()
        self._init_param()
    
    def _init_param(self):
        ckpt_s = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt_s and ckpt_s.model_checkpoint_path:
            self.saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_n_hours=1)
            self.saver.restore(self.sess, ckpt_s.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        self.ckpt_hashing = ray.get(self.ps.get_hashing.remote())
        print("init worker {} checkpoint hashing".format(self.id))
    
    def _data_generator(self):
        global_step = 0
        ep_step, ep_score, d, obs = 0, 0, False, self.env.reset()
        while True:
            # load checkpoint from parameter server if necessary
            if global_step % self.load_ckpt_period == 0:
                ckpt_hashing_ = ray.get(self.ps.get_hashing.remote())
                if ckpt_hashing_ != self.ckpt_hashing:
                    weights = ray.get(self.ps.pull.remote())
                    if weights:
                        self.model.load_from_weights(self.sess, weights)
                        print("worker {} load weights from parameter server, before:{}, after:{}".format(
                            self.id, self.ckpt_hashing, ckpt_hashing_))
                        self.ckpt_hashing = ckpt_hashing_
                    else:
                        print("worker {} load None from parameter server!".format(self.id))
            
            # step in environment
            assert not d and ep_step < self.t_max
            act = self.sess.run(self.model.pi, {self.model.obs: obs.reshape(1,-1)})[0]
            act += self.noise_scale * self.model.act_lim
            act = np.clip(act, -self.model.act_lim, self.model.act_lim)

            obs_, r, d, _ = self.env.step(act)
            ep_step += 1
            ep_score += 1
            global_step += 1
            res = dict(obs=obs, act=act, nex_obs=obs_, r=r, d=d)

            if d or ep_step >= self.t_max:
                ep_step, ep_score, d, obs = 0, 0, False, self.env.reset()
                print("worker {} end episodes, episode step {}, episode score {:.2f}".format(
                    self.id, ep_step, ep_score
                ))
            yield res
    
    def get(self):
        return next(self._data_generator())

class RolloutCollector():
    def __init__(self, ps, **kwargs):
        self.num_workers = int(kwargs['num_workers'])
        self.ps = ps
        self.workers = [
            ray.remote(
                num_cpus=kwargs['cpu_per_worker']
            )(Worker).remote(
                worker_id=i,
                model_fn=build_model,
                env_fn=build_env,
                ps=ps,
                kwargs=kwargs
            ) for i in range(self.num_workers)
        ]
        print("#############################################")
        print("Workers starting ......")
        print("#############################################")

    def _data_id_generator(self, num_returns=1, timeout=None):
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
        
    def get_one_sample(self):
        ready_ids = next(self._data_id_generator())
        return ray_get_and_free(ready_ids)[0]

class BufferReader(Thread):
    def __init__(self, global_buffer, rollout_collector):
        super().__init__()
        self.global_buffer = global_buffer
        self.rollout_collector = rollout_collector

    def run(self):
        while True:
            sample = self.rollout_collector.get_one_sample()
            self.global_buffer.add(sample)

class Learner():
    def __init__(self, model_fn, env_fn, kwargs):
        self.test_env = env_fn(kwargs)

        self.kwargs = kwargs
        self.buffer = OffpolicyReplayBuffer(kwargs['obs_dim'], 
                                            kwargs['act_dim'], 
                                            kwargs['buf_size'])
        # TODO: push into parameter server
        self.ps = AsyncParameterServer.remote()
        self.rollout_collector = RolloutCollector(self.ps, **kwargs)

        self.buffer_reader = BufferReader(global_buffer=self.buffer,
                                          rollout_collector=self.rollout_collector)
        self.buffer_reader.start()

        self.model = model_fn(kwargs)

        backup = tf.stop_gradient(self.model.r + 
                (1-self.model.d) * kwargs['gamma'] * self.model.q_pi_tgt)
        pi_loss = - tf.reduce_mean(self.model.q_pi)
        q_loss = tf.reduce_mean((self.model.q - backup)**2)

        pi_optmizer = tf.train.AdamOptimizer(learning_rate=kwargs['pi_lr'])
        q_optimizer = tf.train.AdamOptimizer(learning_rate=kwargs['q_lr'])

        self.train_pi_op = pi_optmizer.minimize(pi_loss, var_list=get_vars('main/pi'))
        self.train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars("main/q"))

        self.target_update_op = tf.group([tf.assign(tgt_var, 
                                kwargs['rho'] * tgt_var + (1-kwargs['rho']) * var)
                                for var, tgt_var in zip(get_vars('main'), get_vars('target'))])
        
        self.sess = tf.Session()

        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=6)
        ckpt = tf.train.get_checkpoint_state(kwargs['ckpt_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.model.target_init_op)
        
        ray.get(self.ps.push.remote(self.model.get_weights(self.sess)))
    
    def test_once(self):
        ep_score, ep_step, obs = 0, 0, self.test_env.reset()
        d = False
        while not d and ep_step < self.kwargs['t_max']:
            a = self.sess.run(self.model.pi, feed_dict={self.model.obs: obs.reshape(1,-1)})[0]
            obs_, r, d, info = self.test_env.step(a)
            ep_step += 1
            ep_score += r
            obs = obs_
        return ep_score, ep_step
    
    def train(self):
        for t in range(1, self.kwargs['total_steps']+1):
            if t % self.kwargs['update_every'] == 0 and (
                t >= self.kwargs['update_after']) and (
                self.buffer.iter >= kwargs['batch_size']
                ):
                for _ in range(self.kwargs['update_every']):
                    batch = self.buffer.get(self.kwargs['batch_size'])
                    self.sess.run([self.train_pi_op,self.train_q_op],
                        feed_dict=batch)
                    self.sess.run(self.target_update_op)
                
                ray.get(self.ps.push.remote(self.model.get_weights(self.sess)))
            
            scores, steps = [], []
            for _ in range(self.kwargs['update_every']):
                ep_score, ep_step = self.test_once()
                scores.append(ep_score)
                steps.append(ep_step)
            print("test timetep [{}/{}], \t test score {:.2f}, \t test steps {:.2f}".format(
                t, self.kwargs['total_steps'], np.mean(scores), np.mean(steps)
            ))


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Pendulum-v0', 'name of gym environment')
flags.DEFINE_integer('total_steps', int(1e6), 'total steps')
flags.DEFINE_integer('buf_size', int(1e6), "replay buffer size")

flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_float('pi_lr', 1e-3, 'learning rate for policy')
flags.DEFINE_float('q_lr', 1e-3, 'learning rate for q function')

flags.DEFINE_integer('seed', 0, 'random seed')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_float("rho", 0.995, 'smooth factor')

flags.DEFINE_integer('update_after', 1000, 'update after')
flags.DEFINE_integer('update_every', 50, 'update every')

flags.DEFINE_integer("t_max", 200, 'maximum length of 1 episode')
flags.DEFINE_float('noise_scale', 0.01, 'noise scale')

flags.DEFINE_string('ckpt_dir', 'tmp/', 'checkpoint directory')
flags.DEFINE_integer('load_ckpt_period', 20, 'period for worker to load checkpoint')
flags.DEFINE_integer('num_workers', 2, 'number of workers')
flags.DEFINE_integer('cpu_per_worker', 2, 'cpus used for every worker')

kwargs = FLAGS.flag_values_dict()

init_env = build_env(kwargs)
kwargs['obs_dim'] = init_env.observation_space.shape[0]
kwargs['act_dim'] = init_env.action_space.shape[0]
kwargs['act_lim'] = init_env.action_space.high[0]
del init_env

if __name__ == "__main__":
    ray.init()
    learner = Learner(build_training_model, build_env, kwargs)
    learner.train()







        
