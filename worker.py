#!/usr/bin/env python
import cv2  # don't remove it
import go_vncdriver  # don't remove it
import tensorflow as tf
import numpy as np
import argparse
import logging
import sys, signal
import time
import os
from a3c import A3C
from envs import create_env
from model import create_model
from evaluate import evaluate_main
import shutil
import distutils.version

use_tf_0_12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0') and \
                    distutils.version.LooseVersion(tf.VERSION) <= distutils.version.LooseVersion('0.12.1')

use_tf_1_1_api = distutils.version.LooseVersion(tf.VERSION) == distutils.version.LooseVersion('1.1.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)


def run_worker(args, server, cluster):
    env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes)

    max_global_steps = args.max_global_steps
    trainer = A3C(env, args.model_id, args.task, cluster, args.visualise, max_global_steps,
                  lr=args.lr, ent_factor=args.ent_factor,
                  egreedy=args.egreedy, explore_factor=args.explore_factor)

    # Variable names that start with "local" are not saved in checkpoints.
    if use_tf_0_12_api or use_tf_1_1_api:
        variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
    else:
        variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
        init_op = tf.initialize_variables(variables_to_save)
        init_all_op = tf.initialize_all_variables()
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}".format(args.task)],
                            allow_soft_placement=True,
                            log_device_placement=True)
    logdir = os.path.join(args.log_dir, 'train')

    if use_tf_0_12_api or use_tf_1_1_api:
        summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)
    else:
        summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.task)

    logger.info("Events directory: %s_%s", logdir, args.task)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        trainer.start(sess, summary_writer)
        global_step = sess.run(trainer.global_step)
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not max_global_steps or global_step < max_global_steps):
            trainer.process(sess)
            global_step = sess.run(trainer.global_step)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)


def run_worker_val(args):
    val_model_secs = args.val_model_secs
    if val_model_secs < 0:
        logger.info('worker_val: val_model_secs < 0, no validating, exit...')
        return
    logger.info("worker_val: do validating every {} seconds".format(val_model_secs))

    logdir_train = os.path.join(args.log_dir, 'train')
    logdir_val = os.path.join(args.log_dir, 'val')
    logdir_val_tmp = os.path.join(args.log_dir, 'val_tmp')  # TODO: assign a rand name, in case there are racing?
    str_criterion = "score" if args.val_criterion == "score" else "episode length"  # else assume episode_length

    # try to initialize current best
    cur_best = -np.inf
    try:
        score, ep_len = evaluate_main(args.env_val_id, args.model_id, args.max_val_episodes, ckpt_dir=logdir_val,
                                      output_dir="", sleep_time=0.0, render=False, verbose=1)

        cur = score[0] if args.val_criterion == "score" else ep_len[0]  # else assume episode_length
        cur_best = cur if cur> cur_best else cur_best
        logger.info("load initial val model from {}".format(logdir_val))
    except Exception as e:
        logger.info(e)

    logger.info("initial best {} = {}".format(str_criterion, cur_best))

    summary_dir = os.path.join(args.log_dir, 'valid_score')
    if use_tf_0_12_api or use_tf_1_1_api:
        summary_writer = tf.summary.FileWriter(summary_dir)
    else:
        summary_writer = tf.train.SummaryWriter(summary_dir)

    while True:
        time.sleep(val_model_secs)

        def my_rm_dir(dirname):
            def _handler(f, p, err_info):
                logger.info("{} not exists, ignored by rmtree".format(err_info))
            shutil.rmtree(dirname, False, _handler)

        try:
            # backup current checkpoints
            logger.info('validating begins...')
            my_rm_dir(logdir_val_tmp)
            shutil.copytree(logdir_train, logdir_val_tmp)

            # do evaluate
            [score, ep_len], global_step = evaluate_main(args.env_val_id, args.model_id, args.max_val_episodes, ckpt_dir=logdir_val_tmp, output_dir="", sleep_time=0.0, render=False, verbose=1, with_global_step=True)

            logger.info('validating done...')
            with tf.variable_scope("validation"):
                if use_tf_0_12_api or use_tf_1_1_api:
                    tf.summary.scalar('score_avg', score[0])
                    tf.summary.scalar('score_err', score[1])
                    tf.summary.scalar('episode_length_avg', ep_len[0])
                    tf.summary.scalar('episode_length_err', ep_len[1])
                    summary_op = tf.summary.merge_all()
                else:
                    tf.scalar_summary('score_avg', score[0])
                    tf.scalar_summary('score_err', score[1])
                    tf.scalar_summary('episode_length_avg', ep_len[0])
                    tf.scalar_summary('episode_length_err', ep_len[1])
                    summary_op = tf.merge_all_summaries()

            with tf.Session() as sess:
                logger.info('get summary...')
                summary_result = sess.run(summary_op)
                summary_writer.add_summary(summary_result, global_step)

            cur = score[0] if args.val_criterion == "score" else ep_len[0]  # else assume episode_length

            # if better, enforced copy logdir_val_tmp to log_dir_val
            logger.info("current best {} = {}".format(str_criterion, cur_best))
            logger.info("current averaged {} = {}".format(str_criterion, cur))
            if cur > cur_best:
                cur_best = cur
                my_rm_dir(logdir_val)
                shutil.copytree(logdir_val_tmp, logdir_val)
                logger.info('global step: {},\tbest score:{}\n'.format(global_step, cur_best))
                with open(os.path.join(args.log_dir, 'valid_score', 'cur_best.txt'), 'a') as cur_best_file:
                    cur_best_file.write('global step: {},\tbest score:{}\n'.format(global_step, cur_best))
                logger.info("new best model found, saved to {}".format(logdir_val))
            logger.info('validating done')
        except Exception as e:
            logger.info(e)
            continue

    # it never ends, but should be fine. just kill the process when training done


def cluster_spec(num_workers, num_ps):
    """ More tensorflow setup for data parallelism """
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def main(_):
    """ Setting up Tensorflow for data parallel work """

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env-id', default="PongDeterministic-v3", help='Environment id')
    parser.add_argument('--env-val-id', default="Pong-v0", help='Environment id for validation')
    parser.add_argument('--model-id', default="convx2lstm", help='prediction model id')
    parser.add_argument('--max-global-steps', default=100000000, type=int, help='Number of global steps')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--ent-factor', default=0.01, type=float, help='entropy term factor')
    parser.add_argument('--egreedy', default=-1.0, type=float, help='epsilon greedy factor')
    parser.add_argument('--explore-factor', default=1.0, type=float, help='exploration factor')
    parser.add_argument('--val-model-secs', default=-1, type=int, help='Validating model every seconds')
    parser.add_argument('--val-criterion', default="score", help='Validating criterion {score|episode_length}')
    parser.add_argument('--max-val-episodes', default=2, type=int, help='Number of validation episodes')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="tmp/pong", help='Log directory path')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

    # Add visualisation argument
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")

    args = parser.parse_args()
    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128 + signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run_worker(args, server, cluster)
    elif args.job_name == "worker_val":
        run_worker_val(args)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)


if __name__ == "__main__":
    tf.app.run()
