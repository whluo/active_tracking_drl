import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version

use_tf_0_12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0') and \
                    distutils.version.LooseVersion(tf.VERSION) <= distutils.version.LooseVersion('0.12.1')

use_tf_1_1_api = distutils.version.LooseVersion(tf.VERSION) == distutils.version.LooseVersion('1.1.0')

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="VALID", dtype=tf.float32, collections=None):
    """ conv layer, valid padding, init like Torch 7"""
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        bound = 1.0 / np.sqrt(fan_in)

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-bound, bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], dtype, tf.random_uniform_initializer(-bound, bound),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name):
    """ linear layer, init like Torch 7"""
    fan_in = int(x.get_shape()[1])
    bound = 1.0 / np.sqrt(fan_in)
    w_shape = [x.get_shape()[1], size]
    w = tf.get_variable(name + "/w", w_shape, initializer=tf.random_uniform_initializer(-bound, bound))
    b = tf.get_variable(name + "/b", [size], initializer=tf.random_uniform_initializer(-bound, bound))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


class Convx2LSTMActorCritic(object):
    def __init__(self, ob_space, ac_space):
        # screen input
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        # conv block I
        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))

        # conv block II
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))

        #
        x = flatten(x)

        # linear layer
        x = tf.nn.relu(linear(x, 256, 'fc'))

        # add singleton batch dim for LSTM time axis
        x = tf.expand_dims(x, [0])

        # LSTM layer
        size = 256
        if use_tf_1_1_api:
            lstm = rnn.core_rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf_1_1_api:
            state_in = rnn.core_rnn_cell.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # output: policy
        self.logits = linear(x, ac_space, "action")
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # output: value
        self.vf = tf.reshape(linear(x, 1, "value"), [-1])
        # output: LSTM states
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # collect all parameters
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def act_explore(self, ob, c, h):
        return self.act(ob, c, h)

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


class Convx2LSTMActorCriticSmall(object):
    def __init__(self, ob_space, ac_space):
        # screen input
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        # conv block I
        x = tf.nn.relu(conv2d(x, 2, "l1", [8, 8], [4, 4]))

        # conv block II
        x = tf.nn.relu(conv2d(x, 4, "l2", [4, 4], [2, 2]))

        #
        x = flatten(x)

        # linear layer
        x = tf.nn.relu(linear(x, 8, 'fc'))

        # add singleton batch dim for LSTM time axis
        x = tf.expand_dims(x, [0])

        # LSTM layer
        size = 8
        if use_tf_1_1_api:
            lstm = rnn.core_rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf_1_1_api:
            state_in = rnn.core_rnn_cell.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # output: policy
        self.logits = linear(x, ac_space, "action")
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # output: value
        self.vf = tf.reshape(linear(x, 1, "value"), [-1])
        # output: LSTM states
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # collect all parameters
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def act_explore(self, ob, c, h):
        return self.act(ob, c, h)

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


class ScreenVecConvx2LSTMActorCritic(object):
    def __init__(self, ob_screen_space, ob_gamevar_space, ac_space):
        # network input
        self.x = []
        # screen input
        x0 = tf.placeholder(tf.float32, [None] + list(ob_screen_space))
        self.x.append(x0)
        # gamevar input
        x1 = tf.placeholder(tf.float32, [None] + list(ob_gamevar_space))
        self.x.append(x1)

        # conv block I
        x = x0
        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))

        # conv block II
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))

        #
        x = flatten(x)

        # linear layer
        x = tf.nn.relu(linear(x, 256, 'fc'))

        # concat previous feature map x AND gamevar x1
        if use_tf_1_1_api:
            x = tf.concat([x, x1], 1)
        else:
            x = tf.concat(1, [x, x1])

        # add singleton batch dim for LSTM time axis
        x = tf.expand_dims(x, [0])

        # LSTM layer
        size = 256
        if use_tf_1_1_api:
            lstm = rnn.core_rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(x0)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf_1_1_api:
            state_in = rnn.core_rnn_cell.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # output: policy
        self.logits = linear(x, ac_space, "action")
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # output: value
        self.vf = tf.reshape(linear(x, 1, "value"), [-1])
        # output: LSTM states
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # collect all parameters
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample, self.vf] + self.state_out, feed_dict=feed_dict)

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run(self.vf, feed_dict=feed_dict)[0]


class ScreenVecConvx2LSTMActorCriticV2(object):
    def __init__(self, ob_screen_space, ob_gamevar_space, ac_space, explore_factor=1.0):
        # network input
        self.x = []
        # screen input
        x0 = tf.placeholder(tf.float32, [None] + list(ob_screen_space))
        self.x.append(x0)
        # gamevar input
        x1 = tf.placeholder(tf.float32, [None] + list(ob_gamevar_space))
        self.x.append(x1)

        # conv block I
        x = x0
        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))

        # conv block II
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))

        #
        x = flatten(x)

        # linear layer
        x = tf.nn.relu(linear(x, 256, 'fc'))

        # add singleton batch dim for LSTM time axis
        x = tf.expand_dims(x, [0])

        # LSTM layer
        size = 256
        if use_tf_1_1_api:
            lstm = rnn.core_rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(x0)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf_1_1_api:
            state_in = rnn.core_rnn_cell.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # concat previous lstm feature map x AND gamevar x1
        if use_tf_1_1_api:
            x = tf.concat([x, x1], 1)
        else:
            x = tf.concat(1, [x, x1])

        # output: policy
        self.logits = linear(x, ac_space, "action")
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # output: policy with exploration
        self.logits_explore = tf.scalar_mul(explore_factor, self.logits)
        self.sample_explore = categorical_sample(self.logits_explore, ac_space)[0, :]
        # output: value
        self.vf = tf.reshape(linear(x, 1, "value"), [-1])
        # output: LSTM states
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # collect all parameters
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample, self.vf] + self.state_out, feed_dict=feed_dict)

    def act_logits(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample, self.logits, self.vf] + self.state_out, feed_dict=feed_dict)

    def act_explore(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample_explore, self.vf] + self.state_out, feed_dict=feed_dict)

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run(self.vf, feed_dict=feed_dict)[0]


class ScreenVecTensorConvx2LSTMActorCriticV2(object):
    def __init__(self, ob_screen_space, ob_gamevar_space, ob_tensor_space, ac_space, explore_factor=1.0):
        if ob_screen_space[0:2] != ob_tensor_space[0:2]:
            raise Exception("The size of the first two dimension of ob_screen_space and ob_tensor_space MUST be equal")
        # network input
        self.x = []
        # screen input
        x0 = tf.placeholder(tf.float32, [None] + list(ob_screen_space))
        self.x.append(x0)
        # gamevar input
        x1 = tf.placeholder(tf.float32, [None] + list(ob_gamevar_space))
        self.x.append(x1)
        # tensor input (e.g., depth, automap, seglabels)
        x2 = tf.placeholder(tf.float32, [None] + list(ob_tensor_space))
        self.x.append(x2)

        # conv block I
        if use_tf_1_1_api:
            x = tf.concat([x0, x2], 3)
        else:
            x = tf.concat(3, [x0, x2])
        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))

        # conv block II
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))

        #
        x = flatten(x)

        # linear layer
        x = tf.nn.relu(linear(x, 256, 'fc'))

        # add singleton batch dim for LSTM time axis
        x = tf.expand_dims(x, [0])

        # LSTM layer
        size = 256
        if use_tf_1_1_api:
            lstm = rnn.core_rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(x0)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf_1_1_api:
            state_in = rnn.core_rnn_cell.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])

        # concat previous lstm feature map x AND gamevar x1
        if use_tf_1_1_api:
            x = tf.concat([x, x1], 1)
        else:
            x = tf.concat(1, [x, x1])

        # output: policy
        self.logits = linear(x, ac_space, "action")
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # output: policy with exploration
        self.logits_explore = tf.scalar_mul(explore_factor, self.logits)
        self.sample_explore = categorical_sample(self.logits_explore, ac_space)[0, :]
        # output: value
        self.vf = tf.reshape(linear(x, 1, "value"), [-1])
        # output: LSTM states
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # collect all parameters
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample, self.vf] + self.state_out, feed_dict=feed_dict)

    def act_logits(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample, self.logits, self.vf] + self.state_out, feed_dict=feed_dict)

    def act_explore(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample_explore, self.vf] + self.state_out, feed_dict=feed_dict)

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        feed_dict = {self.state_in[0]: c, self.state_in[1]: h}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run(self.vf, feed_dict=feed_dict)[0]


class ScreenVecConvx2ActorCritic(object):
    def __init__(self, ob_screen_space, ob_gamevar_space, ac_space, explore_factor=1.0):
        # network input
        self.x = []
        # screen input
        x0 = tf.placeholder(tf.float32, [None] + list(ob_screen_space))
        self.x.append(x0)
        # gamevar input
        x1 = tf.placeholder(tf.float32, [None] + list(ob_gamevar_space))
        self.x.append(x1)

        # conv block I
        x = x0
        x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4]))

        # conv block II
        x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2]))

        #
        x = flatten(x)

        # linear layer
        x = tf.nn.relu(linear(x, 256, 'fc'))

        # concat previous conv feature map x AND gamevar x1
        if use_tf_1_1_api:
            x = tf.concat([x, x1], 1)
        else:
            x = tf.concat(1, [x, x1])

        # output: policy
        self.logits = linear(x, ac_space, "action")
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        # output: policy with exploration
        self.logits_explore = tf.scalar_mul(explore_factor, self.logits)
        self.sample_explore = categorical_sample(self.logits_explore, ac_space)[0, :]
        # output: value
        self.vf = tf.reshape(linear(x, 1, "value"), [-1])

        # collect all parameters
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        self.state_in = []
        self.state_out = []

    def get_initial_features(self):
        return self.state_in

    def act(self, ob):
        sess = tf.get_default_session()
        feed_dict = {}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample, self.vf] + self.state_out, feed_dict=feed_dict)

    def act_logits(self, ob):
        sess = tf.get_default_session()
        feed_dict = {}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample, self.logits, self.vf] + self.state_out, feed_dict=feed_dict)

    def act_explore(self, ob):
        sess = tf.get_default_session()
        feed_dict = {}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run([self.sample_explore, self.vf] + self.state_out, feed_dict=feed_dict)

    def value(self, ob):
        sess = tf.get_default_session()
        feed_dict = {}
        for xx, obob in zip(self.x, ob):
            feed_dict[xx] = [obob]
        return sess.run(self.vf, feed_dict=feed_dict)[0]


def create_model(model_id, *args):
    if model_id == 'convx2lstm':
        return Convx2LSTMActorCritic(ob_space=args[0], ac_space=args[1])
    if model_id == 'convx2lstm_small':
        return Convx2LSTMActorCriticSmall(ob_space=args[0], ac_space=args[1])
    if model_id == 'screenvec_convx2lstm':
        return ScreenVecConvx2LSTMActorCritic(*args)
    if model_id == 'screenvec_convx2lstm_v2':
        return ScreenVecConvx2LSTMActorCriticV2(*args)
    if model_id == 'screenvectensor_convx2lstm_v2':
        return ScreenVecTensorConvx2LSTMActorCriticV2(*args)
    if model_id == 'screenvec_convx2':
        return ScreenVecConvx2ActorCritic(*args)
    else:
        raise Exception("Unknown model_id {}".format(model_id))
