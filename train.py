import os
import argparse
from datetime import datetime

from algos.sac import SAC
from algos.td3 import TD3
from algos.mocco import MOCCO
from algos.ddpg import DDPG
from trainers.base_trainer import BaseTrainer
from trainers.mocco_trainer import MOCCOTrainer
from env import make_env
from utils import Logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='point_mass-easy', help="Environment name.")
    p.add_argument('--algo', type=str, default='SAC', help="Algorithm name.")
    p.add_argument('--num_steps', type=int, default=int(1e6), help="Total environment steps.")
    p.add_argument("--exp_name", type=str, default="default", help="Experiment name. Used for logging folder naming.")
    p.add_argument("--eval_interval", type=int, default=int(2e3), help="Evaluation interval in environment steps.")
    p.add_argument("--log_every", type=int, default=int(100000), help="Interval for logging progress in stdout.")
    p.add_argument("--beta", type=float, default=0.1, help="MOCCO beta parameter. Not used in other algorithms.")
    p.add_argument('--device', type=str, default="cpu", help="Device to place networks.")
    p.add_argument('--seed', type=int, default=0, help="Random seed.")
    p.add_argument("--logdir", type=str, default="logs", help="Name of the base logging directory.")
    return p.parse_args()


def run(args):
    config = dict(
        env=args.env,
        num_steps=args.num_steps,
        algo=args.algo,
        device=args.device,
        seed=args.seed,
    )

    env = make_env(args.env, seed=args.seed)

    time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    if args.algo == "MOCCO":
        log_dir = os.path.join(
            args.logdir, args.algo, f'{args.algo}_beta_{args.beta}-env_{args.env}-seed{args.seed}-{time}')
    else:
        log_dir = os.path.join(
            args.logdir, args.algo, f'{args.algo}-env_{args.env}-seed{args.seed}-{time}')
    print("LOGDIR: ", log_dir)

    logger = Logger(log_dir)

    STATE_SHAPE = env.observation_space.shape
    ACTION_SHAPE = env.action_space.shape

    print("ALGO: ", args.algo)

    if args.algo == "SAC-tuned":
        trainer_class = BaseTrainer
        algo = SAC(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            tune_alpha=True,
            device=args.device,
            seed=args.seed,
            logger=logger
        )
    elif args.algo == "SAC-fixed":
        trainer_class = BaseTrainer
        algo = SAC(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            tune_alpha=False,
            device=args.device,
            seed=args.seed,
            logger=logger
        )
    elif args.algo == "TD3":
        trainer_class = BaseTrainer
        algo = TD3(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            device=args.device,
            seed=args.seed,
            logger=logger
        )
    elif args.algo == "DDPG":
        trainer_class = BaseTrainer
        algo = DDPG(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            device=args.device,
            seed=args.seed,
            logger=logger
        )
    elif args.algo == "MOCCO":
       trainer_class = MOCCOTrainer
       algo = MOCCO(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            beta=args.beta,
            device=args.device,
            seed=args.seed,
            logger=logger
        )
    else:
        raise ValueError(f"Unsupported algo: {args.algo}")


    def make_test_env(seed: int):
        name = args.env
        return make_env(name, seed)

    trainer = trainer_class(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        env=env,
        make_env_test=make_test_env,
        algo=algo,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        device=args.device,
        log_every=args.log_every,
        seed=args.seed,
        logger=logger,
    )

    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    run(args)
