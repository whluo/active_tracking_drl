import go_vncdriver
from envs import create_env
from gym import wrappers, spaces
from gym.spaces.tuple_space import Tuple
import numpy as np
import time
import argparse
import distutils.version


def avg_err(a):
    avg = a.mean()
    trials = len(a)
    if trials == 1:
        err = 0.0
    else:
        err = np.std(a) / (np.sqrt(trials) - 1)
    return avg, err


def evaluate_loop(env, network, max_episodes, sleep_time, render, verbose):
    """ evaluation for max_episodes

    render: True or False
    verbose: 0 = not print; 1 = basic info; 2 = more info; 3 = even more infos
    """
    last_state = env.reset()
    last_features = network.get_initial_features()
    is_action_logits = isinstance(env.action_space, spaces.Box)
    n_episode, step = 0, 0
    episode_reward = np.zeros((max_episodes,), dtype='float32')
    episode_length = np.zeros((max_episodes,), dtype='float32')

    if verbose >= 1:
        print('evaluating for {} episodes...'.format(max_episodes))
    while True:
        if is_action_logits:
            fetched = network.act_logits(last_state, *last_features)
            action, action_logits, features = fetched[0], fetched[1], fetched[3:]
            state, reward, terminal, _ = env.step(action_logits)
        else:
            fetched = network.act(last_state, *last_features)
            action, features = fetched[0], fetched[2:]
            state, reward, terminal, _ = env.step(action.argmax())

        if render:
            env.render()

        episode_reward[n_episode] += reward
        if verbose >= 3:
            if reward >= 0.001 or reward <= -0.001:
                print("#step = {}, action = {}, reward = {}".format(step, action.argmax(), reward))

        if verbose >= 4:
            print("#step = {}, action = {}, reward = {}".format(step, action.argmax(), reward))

        if terminal:
            if verbose >= 2:
                print("#episode = {}, #step = {}, reward sum = {}".format(n_episode, step, episode_reward[n_episode]))
            episode_length[n_episode] = step

            step = 0
            n_episode += 1
            if n_episode >= max_episodes:
                break

            last_state = env.reset()
            last_features = network.get_initial_features()
        else:
            last_state = state
            last_features = features

            step += 1
            if sleep_time >= 0:
                time.sleep(sleep_time)
            else:
                raw_input('press any key to proceed')  # wait keypress

    s_avg, s_err = avg_err(episode_reward)
    l_avg, l_err = avg_err(episode_length)
    if verbose >= 1:
        print('evaluation done.')
        print('scores = {} +- {}'.format(s_avg, s_err))
        print('episode length = {} +- {}'.format(l_avg, l_err))

    return (s_avg, s_err), (l_avg, l_err)


def evaluate_main(env_id, model_id, max_episodes, ckpt_dir, output_dir,
                  sleep_time, render, verbose, with_global_step=False):
    # env
    env = create_env(env_id, 0, 1)
    if len(output_dir) > 0:  # output recording
        env = wrappers.Monitor(env, output_dir)
    if render:
        env.render()

    is_obs_tuple = isinstance(env.observation_space, Tuple)
    observation_shape = [sp.shape for sp in env.observation_space.spaces] if is_obs_tuple \
        else [env.observation_space.shape]
    action_shape = [env.action_space.n] if isinstance(env.action_space, spaces.Discrete) \
        else [env.action_space.shape[0]]

    # work-around to the nasty env.render() failing issue when working with tensorflow
    # see https://github.com/openai/gym/issues/418
    import tensorflow as tf
    from model import create_model
    use_tf_0_12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0') and \
                    distutils.version.LooseVersion(tf.VERSION) <= distutils.version.LooseVersion('0.12.1')
    use_tf_1_1_api = distutils.version.LooseVersion(tf.VERSION) == distutils.version.LooseVersion('1.1.0')

    # model
    tf.reset_default_graph()
    sess = tf.Session()
    with tf.variable_scope("global"):
        network = create_model(model_id, *observation_shape + action_shape)
        if(with_global_step):
            global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)

    init = tf.global_variables_initializer()
    sess.run(init)


    # load model parameters
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    if checkpoint and checkpoint.model_checkpoint_path:

        restore_tf_0_12_model = False
        restore_tf_1_1_model = False

        reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
        for var_name in reader.get_variable_to_shape_map():
            if 'RNN/BasicLSTMCell/Linear' in var_name:
                restore_tf_0_12_model = True
                break
            elif 'rnn/basic_lstm_cell/' in var_name:
                restore_tf_1_1_model = True
                break

        if use_tf_1_1_api and restore_tf_0_12_model:
            var_dict = {}
            for var in tf.global_variables():
                name = var.name.split(':')[0]
                if 'rnn/basic_lstm_cell/weights' in name:
                    name = name.replace('rnn/basic_lstm_cell/weights', 'RNN/BasicLSTMCell/Linear/Matrix')
                elif 'rnn/basic_lstm_cell/biases' in name:
                    name = name.replace('rnn/basic_lstm_cell/biases', 'RNN/BasicLSTMCell/Linear/Bias')
                var_dict[name] = var
            saver = tf.train.Saver(var_dict)
        elif use_tf_0_12_api and restore_tf_1_1_model:
            var_dict = {}
            for var in tf.global_variables():
                name = var.name.split(':')[0]
                if 'RNN/BasicLSTMCell/Linear/Matrix' in name:
                    name = name.replace('RNN/BasicLSTMCell/Linear/Matrix', 'rnn/basic_lstm_cell/weights')
                elif 'RNN/BasicLSTMCell/Linear/Bias' in name:
                    name = name.replace('RNN/BasicLSTMCell/Linear/Bias', 'rnn/basic_lstm_cell/biases')
                var_dict[name] = var
            saver = tf.train.Saver(var_dict)
        else:
            saver = tf.train.Saver()
    
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
        raise Exception('cannot find checkpoint path')

    # run evaluating
    with sess.as_default():
        ret = evaluate_loop(env, network, max_episodes, sleep_time, render, verbose)
        env.close()
        if(with_global_step):
            global_step_result = sess.run(global_step)
    sess.close()

    if(with_global_step):
        return ret, global_step_result
    else:
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env-id', default="BreakoutDeterministic-v3", help='Environment id')
    parser.add_argument('--model-id', default="convx2lstm", help='Model id')
    parser.add_argument('--ckpt-dir', default="save/breakout/train", help='Checkpoint directory path')
    parser.add_argument('--output-dir', default="", help='Output directory path')
    parser.add_argument('--max-episodes', default=2, type=int, help='Number of episodes to evaluate')
    parser.add_argument('--sleep-time', default=0.0, type=float, help='sleeping time. -1 for waiting keypress')
    parser.add_argument('--render', action='store_true', help='render screen')
    parser.add_argument('--verbose', default=3, type=int, help='verbose. {0, 1, 2, 3, 4}. 0 means silent ')

    args = parser.parse_args()
    evaluate_main(args.env_id, args.model_id, args.max_episodes, args.ckpt_dir, args.output_dir,
                  args.sleep_time, args.render, args.verbose)
