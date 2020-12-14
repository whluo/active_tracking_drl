from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from gym import spaces
from gym.spaces.tuple_space import Tuple
from model import create_model
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
from random import random, randint

use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')


def anneal_rate(t, max_t, init_rate):
    lr = init_rate * (max_t - t) / max_t
    if lr < 0.0:
        lr = 0.0
    return lr


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, is_state_tuple, gamma, lambda_=1.0):
    """ given a rollout, compute its returns and the advantage """
    if is_state_tuple:
        # "transpose". For example, [[s1, ss1], [s2, ss2], [s3, ss3]] --> [(s1, s2, s3), (ss1, ss2, ss3)]
        list_of_state_rollout = map(list, zip(*rollout.states))
        batch_si = [np.asarray(cur_state_rollout) for cur_state_rollout in list_of_state_rollout]
    else:
        batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)


Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])


class PartialRollout(object):
    """ a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps. """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


class RunnerThread(threading.Thread):
    """ One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """

    def __init__(self, env, policy, num_local_steps, visualise, egreedy=0.0):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.egreedy = egreedy

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise,
                                      self.egreedy)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(env, policy, num_local_steps, summary_writer, render, egreedy=-1.0):
    """ The logic of the thread runner.

    In brief, it constantly keeps on running the policy, and as long as the rollout exceeds a certain length,
    the thread runner appends the policy to the queue. """

    last_state = env.reset()
    last_features = policy.get_initial_features()
    episode_length = 0
    rewards = 0
    is_egreedy = egreedy > 0.0
    is_ac_vec = isinstance(env.action_space, spaces.Box)

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for _ in range(num_local_steps):
            if is_ac_vec:
                # ignore exploration rate
                fetched = policy.act_logits(last_state, *last_features)
                action, action_logits, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]
                # pass distribution, ignore egreedy
                state, reward, terminal, info = env.step(action_logits)
            else:
                # take action with exploration
                fetched = policy.act_explore(last_state, *last_features)
                action, value_, features = fetched[0], fetched[1], fetched[2:]
                # choose action egreedy
                if is_egreedy and random() < egreedy:
                    action_index = randint(0, len(action) - 1)
                    action = np.zeros_like(action)
                    action[action_index] = 1
                else:
                    action_index = action.argmax()
                state, reward, terminal, info = env.step(action_index)

            if render:
                env.render()

            # collect the experience
            reward_clipped = np.clip(reward, -1.0, +1.0)
            rollout.add(last_state, action, reward_clipped, value_, terminal, last_features)
            episode_length += 1
            rewards += reward

            last_state = state
            last_features = features

            if info:
                summary = tf.Summary()
                for k, v in info['to_log'].items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            # print('timestep_limit: ',timestep_limit)
            # print('terminal: ', terminal)
            if terminal or episode_length >= timestep_limit:
                # print('episode_length: ', episode_length)
                terminal_end = True
                # print('terminal_end: ', terminal_end)
                if episode_length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_features = policy.get_initial_features()
                # print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, episode_length))
                episode_length = 0
                rewards = 0
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout



class A3C(object):
    def __init__(self, env, model_id, task, cluster, visualise, max_global_steps, lr=1e-4, ent_factor=0.01, egreedy=-0.1,
                 explore_factor=0.0):
        """ An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.

        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed. """

        self.env = env
        self.task = task
        self.max_global_steps = max_global_steps

        self.is_obs_tuple = isinstance(env.observation_space, Tuple)
        self.is_ac_vec = isinstance(env.action_space, spaces.Box)

        observation_shape = [sp.shape for sp in env.observation_space.spaces] if self.is_obs_tuple \
            else [env.observation_space.shape]
        action_shape = [env.action_space.shape[0]] if self.is_ac_vec else [env.action_space.n]

        worker_device = "/job:worker/task:{}".format(task)
        with tf.device(tf.train.replica_device_setter(cluster=cluster, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = create_model(model_id, *observation_shape + action_shape + [explore_factor])
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.local_network = pi = create_model(model_id, *observation_shape + action_shape + [explore_factor])
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None] + action_shape, name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            self.loss = pi_loss + 0.5 * vf_loss - entropy * ent_factor

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            # self.runner = RunnerThread(env, pi, 20, visualise, egreedy=egreedy)
            self.runner = RunnerThread(env, pi, 200, visualise, egreedy=egreedy)

            grads = tf.gradients(self.loss, pi.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            if self.is_obs_tuple:
                x_screen = pi.x[0]
            else:
                x_screen = pi.x
            is_x_image = int(x_screen.get_shape()[-1]) in [1, 3, 4]

            x_visible_tensor = []
            if self.is_obs_tuple and len(pi.x) > 2:
                for i in range(2, len(pi.x)):
                    if int(pi.x[i].get_shape()[-1]) in [1, 3, 4]:
                        x_visible_tensor.append(pi.x[i])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(x_screen)[0])            

            # the optimizer. each worker has its own
            opt = tf.train.AdamOptimizer(lr)
            self.lr = tf.placeholder(tf.float32)
            # opt = tf.train.RMSPropOptimizer(self.lr, decay=0.99, momentum=0.0, epsilon=0.1, use_locking=False)
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0
            

            # summary
            bs = tf.to_float(tf.shape(x_screen)[0])
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                if is_x_image:
                    tf.summary.image("model/state", x_screen)
                if len(x_visible_tensor) > 0:
                    for i in range(len(x_visible_tensor)):
                        tf.summary.image("model/visible_tensor_" + str(i), x_visible_tensor[i])
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                tf.summary.scalar("model/lr", self.lr)
                self.summary_op = tf.summary.merge_all()

            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                if is_x_image:
                    tf.image_summary("model/state", x_screen)
                if len(x_visible_tensor) > 0:
                    for i in range(len(x_visible_tensor)):
                        tf.summary.image("model/visible_tensor_" + str(i), x_visible_tensor[i])
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                tf.scalar_summary.scalar("model/lr", self.lr)
                self.summary_op = tf.merge_all_summaries()

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """ self explanatory:  take a rollout from the queue of the thread runner. """

        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """ process grabs a rollout that's been produced by the thread runner,
        and updates the parameters. The update is then sent to the parameter server. """

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, self.is_obs_tuple, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        cur_global_step = self.global_step.eval()

        feed_dict = {
            self.lr: anneal_rate(cur_global_step, self.max_global_steps, 0.0007),
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
        }
        for i in range(len(batch.features)):
            feed_dict[self.local_network.state_in[i]] = batch.features[i]
        if self.is_obs_tuple:
            for xx, ss in zip(self.local_network.x, batch.si):
                feed_dict[xx] = ss
        else:
            feed_dict[self.local_network.x] = batch.si

        fetched = sess.run(fetches, feed_dict=feed_dict)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()
        self.local_steps += 1
