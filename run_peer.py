import argparse
import datetime

import gym

from pathlib import Path

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.utils import set_random_seed, \
    update_learning_rate

import wandb
from wandb.integration.sb3 import WandbCallback

import predefined_agents  # noqa: F401
from dqn_peer import DQNPeer
from peer import PeerGroup, make_peer_class
import env as local_envs  # noqa: F401

from callbacks import PeerEvalCallback

from utils import str2bool, add_default_values_to_parser, \
    log_reward_avg_in_wandb, add_default_values_to_train_parser, \
    new_random_seed, make_env, ControllerArguments


def add_args():
    # create arg parser
    parser = argparse.ArgumentParser(description="Peer learning.")
    # General
    parser.add_argument("--save-name", type=str, default="delete_me")
    parser = add_default_values_to_parser(parser)

    # Training
    training = parser.add_argument_group("Training")
    add_default_values_to_train_parser(training)

    # Peer Learning
    peer_learning = parser.add_argument_group("Peer Learning")
    peer_learning.add_argument("--follow-steps", type=int, default=10)
    peer_learning.add_argument("--switch-ratio", type=float, default=1,
                               help="How many times peer training compared to "
                                    "solo training Ratio of peer learning "
                                    "episodes to solo episodes; 0 -> only "
                                    "peer learning episodes."
                                    "ratio 0 {'solo': 0, 'peer': 100}"
                                    "ratio 0.2 {'solo': 83, 'peer': 17}"
                                    "ratio 0.25 {'solo': 80, 'peer': 20}"
                                    "ratio 0.333333 {'solo': 75, 'peer': 25}"
                                    "ratio 0.5 {'solo': 67, 'peer': 33}"
                                    "ratio 1 {'solo': 50, 'peer': 50}"
                                    "ratio 2 {'solo': 33, 'peer': 67}"
                                    "ratio 3 {'solo': 25, 'peer': 75}"
                                    "ratio 4 {'solo': 20, 'peer': 80}"
                                    "ratio 5 {'solo': 17, 'peer': 83}")
    peer_learning.add_argument("--peer-learning", type=str2bool, nargs="?",
                               const=True, default=True)
    peer_learning.add_argument("--peers-sample-with-noise", type=str2bool,
                               nargs="?",
                               const=True, default=True)
    peer_learning.add_argument("--use-agent-value", type=str2bool, nargs="?",
                               const=True, default=True)
    peer_learning.add_argument("--use-trust", type=str2bool, nargs="?",
                               const=True, default=True)
    peer_learning.add_argument("--use-trust-buffer", type=str2bool, nargs="?",
                               const=True, default=True)
    peer_learning.add_argument("--trust-buffer-size", type=int, default=1000)
    peer_learning.add_argument("--use-critic", type=str2bool, nargs="?",
                               const=True, default=True)
    peer_learning.add_argument("--sample_random_actions", type=str2bool,
                               nargs="?", const=True, default=False)
    peer_learning.add_argument("--trust-lr", type=float, default=0.001)
    peer_learning.add_argument("--T", type=float, nargs='*', default=[1])
    peer_learning.add_argument("--T-decay", type=float, nargs='*', default=[0])
    peer_learning.add_argument("--init-trust-values", type=float, default=200)
    peer_learning.add_argument("--init-agent-values", type=float, default=200)
    peer_learning.add_argument("--use-advantage", type=str2bool, nargs="?",
                               const=False, default=False)
    peer_learning.add_argument("--sample-from-suggestions", type=str2bool,
                               nargs="?", const=False, default=False)
    peer_learning.add_argument("--epsilon", type=float, default=0.0)
    peer_learning.add_argument("--max-peer-epochs", type=int,
                               default=1_000_000_000)
    peer_learning.add_argument("--only-follow-peers", type=str2bool,
                               nargs="?", const=False, default=False)

    return parser


if __name__ == '__main__':
    # parse args
    arg_parser = add_args()
    args = arg_parser.parse_args()
    CA = ControllerArguments(args.agent_count)

    # assert if any peer learning strategy is chosen peer learning must be True
    option_on = (args.use_trust or args.use_critic or args.use_agent_value)
    assert (option_on and args.peer_learning) or not option_on

    # create results/experiments folder
    time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    unique_dir = f"{time_string}__{args.job_id}"
    experiment_folder = args.save_dir.joinpath(args.save_name, unique_dir)
    experiment_folder.mkdir(exist_ok=True, parents=True)
    str_folder = str(experiment_folder)
    print("Experiment folder is", str_folder)

    # suppress gym warnings
    gym.logger.set_level(level=gym.logger.DISABLED)

    # seed everything
    set_random_seed(args.seed)

    # init wandb
    wandb.tensorboard.patch(root_logdir=str_folder)
    run = wandb.init(entity="jgu-wandb", config=args.__dict__,
                     project="peer-learning",
                     monitor_gym=True, sync_tensorboard=False,
                     name=f"{args.save_name}__{args.job_id}",
                     notes=f"Peer Learning with {args.agent_count} agents on "
                           f"the {args.env.split('-')[0]} environment.",
                     dir=str_folder, mode=args.wandb)

    # initialize peer group
    algo_args = []
    peer_args = []
    for i in range(args.agent_count):
        algo_args.append(
            dict(policy="MlpPolicy",
                 verbose=1,
                 policy_kwargs=dict(
                     net_arch=CA.argument_for_every_agent(args.net_arch, i)
                 ),
                 buffer_size=args.buffer_size,
                 batch_size=args.batch_size,
                 gamma=args.gamma,
                 tau=args.tau,
                 train_freq=args.train_freq,
                 target_update_interval=args.target_update_interval,
                 gradient_steps=args.gradient_steps,
                 learning_starts=args.buffer_start_size,
                 learning_rate=CA.argument_for_every_agent(args.learning_rate,
                                                           i),
                 tensorboard_log=None,
                 device=args.device))

        peer_args.append(
            dict(temperature=CA.argument_for_every_agent(args.T, i),
                 temp_decay=CA.argument_for_every_agent(args.T_decay, i),
                 algo_args=algo_args[i],
                 env=args.env,
                 env_args=args.env_args,
                 use_trust=args.use_trust,
                 use_critic=args.use_critic,
                 buffer_size=args.trust_buffer_size,
                 follow_steps=args.follow_steps,
                 use_trust_buffer=args.use_trust_buffer,
                 solo_training=not args.peer_learning,
                 peers_sample_with_noise=args.peers_sample_with_noise,
                 sample_random_actions=args.sample_random_actions,
                 init_trust_values=args.init_trust_values,
                 sample_from_suggestions=args.sample_from_suggestions,
                 epsilon=args.epsilon,
                 only_follow_peers=args.only_follow_peers))

    # create Peer classes
    SACPeer = make_peer_class(SAC)
    TD3Peer = make_peer_class(TD3)

    # create peers and peer group
    peers = []
    callbacks = []
    eval_envs = []
    for i in range(args.agent_count):
        args_for_agent = peer_args[i]
        agent_algo = CA.argument_for_every_agent(args.mix_agents, i)

        if agent_algo == 'SAC':
            args_for_agent["algo_args"]["ent_coef"] = "auto"
            args_for_agent["algo_args"]["use_sde"] = True
            args_for_agent["algo_args"]["policy_kwargs"]["log_std_init"] = -3
            peer = SACPeer(**args_for_agent, seed=new_random_seed())

        elif agent_algo == 'TD3':
            peer = TD3Peer(**args_for_agent, seed=new_random_seed())

        elif agent_algo == 'DQN':
            args_for_agent["algo_args"]["exploration_fraction"] = \
                args.exploration_fraction
            args_for_agent["algo_args"]["exploration_final_eps"] = \
                args.exploration_final_eps
            peer = DQNPeer(**args_for_agent, seed=new_random_seed())

        elif agent_algo in ['Adversarial', 'Expert']:
            class_str = f"predefined_agents." \
                        f"{args.env.split('-')[0]}{agent_algo}"
            peer = eval(class_str)(**args_for_agent, seed=new_random_seed())
        else:
            raise NotImplementedError(
                f"The Agent {agent_algo}"
                f" is not implemented")
        peers.append(peer)

        eval_env = make_env(args.env, args.n_eval_episodes, **args.env_args)
        # every agent gets its own callbacks
        callbacks.append([WandbCallback(verbose=2)])
        eval_envs.append(eval_env)

    peer_group = PeerGroup(peers, use_agent_values=args.use_agent_value,
                           lr=args.trust_lr, switch_ratio=args.switch_ratio,
                           init_agent_values=args.init_agent_values,
                           use_advantage=args.use_advantage,
                           max_peer_epochs=args.max_peer_epochs)

    # create callbacks
    for i in range(args.agent_count):
        peer_callback = PeerEvalCallback(eval_env=eval_envs[i],
                                         eval_envs=eval_envs,
                                         peer_group=peer_group,
                                         best_model_save_path=str_folder,
                                         log_path=str_folder,
                                         eval_freq=args.eval_interval,
                                         n_eval_episodes=args.n_eval_episodes)
        callbacks[i].append(peer_callback)  # type: ignore

    # calculate number of epochs based on episode length
    max_episode_steps = max(args.min_epoch_length,
                            gym.spec(args.env).max_episode_steps)

    n_epochs = args.steps // max_episode_steps
    # load pretrained model
    for i, path in enumerate(args.load_paths):
        load_path = Path.cwd().joinpath("Experiments", path)
        peer = peer_group.peers[i].set_parameters(load_path_or_dict=load_path)
        peers[i].learning_rate = 0
        peers[i].lr_schedule = lambda _: 0.0
        update_learning_rate(peers[i].ent_coef_optimizer, 0)
        peers[i].replay_buffer.reset()
        peers[i].buffer.buffer.clear()

    # train the peer group
    peer_group.learn(n_epochs, callbacks=callbacks,
                     eval_log_path=str_folder,
                     max_epoch_len=max_episode_steps)

    log_reward_avg_in_wandb(callbacks)

    for i in args.agents_to_store:
        peers[i].save(path=experiment_folder / f'trained_model_{i}')
