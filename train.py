import argparse
import os
import sys
from six.moves import shlex_quote

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-r', '--remotes', default=None,
                    help='The address of pre-existing VNC servers and '
                         'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901).')
parser.add_argument('--env-id', type=str, default="PongDeterministic-v3",
                    help="Environment id")
parser.add_argument('--env-val-id', type=str, default="",
                    help="Environment id for validation. Use training env if null string")
parser.add_argument('--model-id', default="convx2lstm",
                    help='Number of global steps')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--ent-factor', default=0.01, type=float,
                    help='entropy term factor')
parser.add_argument('--egreedy', default=-1.0, type=float,
                    help='epsilon greedy factor')
parser.add_argument('--explore-factor', default=1.0, type=float,
                    help='exploration factor')
parser.add_argument('--max-global-steps', default=100000000, type=int,
                    help='Number of global steps')
parser.add_argument('--max-val-episodes', default=3, type=int,
                    help='Number of validating episodes')
parser.add_argument('--val-model-secs', default=-1, type=int,
                    help='Validating model every seconds')
parser.add_argument('--val-criterion', default="score",
                    help='Validating criterion {score|episode_length}')
parser.add_argument('-l', '--log-dir', type=str, default="tmp/pong",
                    help="Log directory path")
parser.add_argument('-n', '--dry-run', action='store_true',
                    help="Print out commands rather than executing them")
parser.add_argument('--sleep-worker', default=10, type=float,
                    help='sleeping time after starting a worker (before starting the next worker)')
parser.add_argument('-m', '--mode', type=str, default='tmux',
                    help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")


# Add visualise tag
parser.add_argument('--visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")


def new_cmd(session, name, cmd, mode, logdir, shell):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)
    if mode == 'tmux':
        return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))
    elif mode == 'child':
        return name, "{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(cmd, logdir, session, name, logdir)
    elif mode == 'nohup':
        return name, "nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(shell, shlex_quote(cmd),
                                                                                            logdir, session, name,
                                                                                            logdir)


def create_commands(session, num_workers, remotes, env_id, model_id, logdir, shell='bash', mode='tmux', visualise=False,
                    max_global_steps=100000000, val_model_secs=60, val_criterion='score', max_val_episodes=1, lr=1e-4,
                    ent_factor=0.01, egreedy=-1.0, explore_factor=1.0, env_val_id='', sleep_worker=0.0):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        # 'CUDA_VISIBLE_DEVICES=',
        sys.executable, 'worker.py',
        '--log-dir', logdir,
        '--env-id', env_id,
        '--model-id', model_id,
        '--num-workers', str(num_workers),
        '--max-global-steps', max_global_steps,
        '--lr', lr,
        '--ent-factor', ent_factor,
        '--egreedy', egreedy,
        '--explore-factor', explore_factor,
    ]

    if visualise:
        base_cmd += ['--visualise']

    if remotes is None:
        remotes = ["1"] * num_workers
    else:
        remotes = remotes.split(',')
        assert len(remotes) == num_workers

    # ps
    cmds_map = [new_cmd(session, "ps", ['CUDA_VISIBLE_DEVICES='] + base_cmd + ["--job-name", "ps"], mode, logdir, shell)]

    # workers for training
    for i in range(num_workers):
        cmds_map += [new_cmd(session, "w-%d" % i,
                             ['CUDA_VISIBLE_DEVICES=0'] + base_cmd + ["--job-name", "worker", "--task", str(i), "--remotes", remotes[i]],
                             mode, logdir, shell)]

    # worker for validation
    if not env_val_id:
        env_val_id = env_id  # use training env
    cmds_map += [new_cmd(session, "w-val",
                         base_cmd + ["--job-name", "worker_val", "--task", str(num_workers),
                                     "--val-model-secs", str(val_model_secs),
                                     "--max-val-episodes", str(max_val_episodes),
                                     "--val-criterion", val_criterion,
                                     "--env-val-id", env_val_id],
                         mode, logdir, shell)]

    # tensorboard
    cmds_map += [new_cmd(session, "tb", ["tensorboard", "--logdir", logdir, "--port", "12345"], mode, logdir, shell)]

    # htop watcher
    if mode == 'tmux':
        cmds_map += [new_cmd(session, "htop", ["htop"], mode, logdir, shell)]

    windows = [v[0] for v in cmds_map]

    notes = []
    cmds = [
        "mkdir -p {}".format(logdir),
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']),
                                        logdir),
    ]
    if mode == 'nohup' or mode == 'child':
        cmds += ["echo '#!/bin/sh' >{}/kill.sh".format(logdir)]
        notes += ["Run `source {}/kill.sh` to kill the job".format(logdir)]
    if mode == 'tmux':
        notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
        notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]
    else:
        notes += ["Use `tail -f {}/*.out` to watch process output".format(logdir)]
    notes += ["Point your browser to http://localhost:12345 to see Tensorboard"]

    if mode == 'tmux':
        cmds += [
            "kill $( lsof -i:12345 -t ) > /dev/null 2>&1",  # kill any process using tensorboard's port
            "kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1".format(num_workers + 12222),  # kill
            "tmux kill-session -t {}".format(session),
            "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
        ]
        for w in windows[1:]:
            cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
        cmds += ["sleep 1"]
    for window, cmd in cmds_map:
        cmds += [cmd]
        if window[0] == 'w':
            cmds += ["sleep {}".format(sleep_worker)]

    return cmds, notes


def run():
    args = parser.parse_args()
    cmds, notes = create_commands("a3c", args.num_workers, args.remotes, args.env_id, args.model_id, args.log_dir,
                                  mode=args.mode, visualise=args.visualise, max_global_steps=args.max_global_steps,
                                  val_model_secs=args.val_model_secs, max_val_episodes=args.max_val_episodes,
                                  val_criterion=args.val_criterion, lr=args.lr, ent_factor=args.ent_factor,
                                  egreedy=args.egreedy, explore_factor=args.explore_factor, env_val_id=args.env_val_id,
                                  sleep_worker=args.sleep_worker)
    if args.dry_run:
        print("Dry-run mode due to -n flag, otherwise the following commands would be executed:")
    else:
        print("Executing the following commands:")
    print("\n".join(cmds))
    print("")
    if not args.dry_run:
        if args.mode == "tmux":
            os.environ["TMUX"] = ""
        os.system("\n".join(cmds))
    print('\n'.join(notes))


if __name__ == "__main__":
    run()
