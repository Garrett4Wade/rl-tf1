import tensorflow as tf
import numpy as np
import gym

from replay_buffer import OffpolicyReplayBuffer

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
        q_pi = tf.squeeze(mlp(tf.concat([obs, pi], -1), list(hidden_size)+[1], activation), -1)
    return pi, q, q_pi

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

class Agent():
    def __init__(self, env_fn, FLAGS):
        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)
        self.flags = FLAGS

        self.env, self.test_env = env_fn(), env_fn()

        self.action_space = self.env.action_space
        self.act_dim = self.action_space.shape[0]
        self.obs_space = self.env.observation_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_lim = self.action_space.high[0]

        self.obs_ph, self.a_ph, self.nex_obs_ph, self.r_ph, self.d_ph = \
            tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim)), \
            tf.placeholder(dtype=tf.float32, shape=(None, self.act_dim)), \
            tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim)), \
            tf.placeholder(dtype=tf.float32, shape=(None,)), \
            tf.placeholder(dtype=tf.float32, shape=(None,))

        with tf.variable_scope("main"):
            self.pi, q, pi_q = ddpg_mlp_actor_critic(self.obs_ph, self.a_ph, self.act_dim, self.act_lim)
        
        with tf.variable_scope("target"):
            _, _, pi_q_tgt = ddpg_mlp_actor_critic(self.nex_obs_ph, self.a_ph, self.act_dim, self.act_lim)
        
        self.buffer = OffpolicyReplayBuffer(self.obs_dim, 
                                            self.act_dim, 
                                            self.flags.buf_size)

        backup = tf.stop_gradient(self.r_ph + (1-self.d_ph) * self.flags.gamma * pi_q_tgt)
        self.pi_loss = - tf.reduce_mean(pi_q)
        self.q_loss = tf.reduce_mean((q - backup)**2)

        self.pi_optmizer = tf.train.AdamOptimizer(learning_rate=self.flags.pi_lr)
        self.q_optimizer = tf.train.AdamOptimizer(learning_rate=self.flags.q_lr)

        self.train_pi_op = self.pi_optmizer.minimize(self.pi_loss, var_list=get_vars('main/pi'))
        self.train_q_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars("main/q"))

        self.target_update_op = tf.group([tf.assign(tgt_var, 
                                self.flags.rho * tgt_var + (1-self.flags.rho) * var)
                                for var, tgt_var in zip(get_vars('main'), get_vars('target'))])
        self.target_init_op = tf.group([tf.assign(tgt_var, var)
                                for var, tgt_var in zip(get_vars("main"), get_vars("target"))])
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init_op)
    
    def act(self, obs):
        pure_a = self.sess.run(self.pi, feed_dict={self.obs_ph: obs.reshape(1,-1)})[0]
        return np.clip(pure_a + self.flags.noise_scale * self.act_lim,
                            -self.act_lim, self.act_lim)
    
    def test_once(self):
        ep_score, ep_step, obs = 0, 0, self.test_env.reset()
        d = False
        while not d and ep_step < self.flags.max_t:
            a = self.sess.run(self.pi, feed_dict={self.obs_ph: obs.reshape(1,-1)})[0]
            obs_, r, d, info = self.test_env.step(a)
            ep_step += 1
            ep_score += r
            obs = obs_
        return ep_score, ep_step
    
    def train(self):
        ep_score, ep_step, d, obs = 0, 0, False, self.env.reset()
        for i in range(self.flags.total_steps):
            global_step = i + 1
            a = self.act(obs)
            obs_, r, d, info = self.env.step(a)
            ep_score += r
            ep_step += 1

            self.buffer.add(obs, a, obs_, r, d)

            obs = obs_

            if d or ep_step >= self.flags.max_t:
                ep_score, ep_step, d, obs = 0, 0, False, self.env.reset()
            
            if (global_step > self.flags.update_after 
                    and global_step % self.flags.update_every == 0):
                for _ in range(self.flags.update_every):
                    batch = self.buffer.get(self.flags.batch_size)
                    fd = {
                        self.obs_ph: batch['obs'],
                        self.a_ph: batch['act'],
                        self.nex_obs_ph: batch['nex_obs'],
                        self.r_ph: batch['r'],
                        self.d_ph: batch['d']
                    }
                    q_loss, pi_loss, _, _ = self.sess.run(
                        [self.q_loss, self.pi_loss, self.train_q_op, self.train_pi_op], 
                        fd)
                self.sess.run(self.target_update_op)
                
                scores, steps = [], []
                for _ in range(self.flags.update_every):
                    ep_score, ep_step = self.test_once()
                    scores.append(ep_score)
                    steps.append(ep_step)
                print("timetep [{}/{}], \t test score {:.2f}, \t test steps {:.2f}".format(
                    global_step, self.flags.total_steps, np.mean(scores), np.mean(steps)
                ))

if __name__ == "__main__":
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('env_name', 'Pendulum-v0', 'name of gym environment')
    flags.DEFINE_integer('total_steps', int(1e6), 'total steps')
    flags.DEFINE_integer('buf_size', int(1e6), "replay buffer size")

    flags.DEFINE_integer('batch_size', 64, 'batch size')
    flags.DEFINE_float('pi_lr', 1e-3, 'learning rate for policy')
    flags.DEFINE_float('q_lr', 1e-3, 'learning rate for q function')

    flags.DEFINE_integer('seed', 0, 'random seed')
    flags.DEFINE_float('gamma', 0.99, 'discount factor')
    flags.DEFINE_float("rho", 0.995, 'smooth factor')
    
    flags.DEFINE_integer('update_after', 1000, 'update after')
    flags.DEFINE_integer('update_every', 10, 'update every')
    
    flags.DEFINE_integer("max_t", 200, 'maximum length of 1 episode')
    flags.DEFINE_float('noise_scale', 0.01, 'noise scale')

    env_fn = lambda: gym.make(FLAGS.env_name)
    agent = Agent(env_fn, FLAGS)
    agent.train()
        
        
