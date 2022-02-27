import os
import argparse
from datetime import datetime

import wandb

from algos.sac import SAC
from algos.td3 import TD3
from algos.ddpg import DDPG
from algos.mocco import MOCCO
from trainers.gembo_trainer import GemboTrainer
from trainers.mf_trainer import ModelFreeTrainer
from env import make_env


def run(args):
    config = dict(
        env=args.env,
        num_steps=args.num_steps,
        algo=args.algo,
        device=args.device,
        seed=args.seed,
    )
    wandb.init(project="MOCCO", config=config, group=f"{args.env}-{args.algo}-{args.exp_name}")

    env = make_env(args.env, args.seed)
    env_test = make_env(args.env, args.seed)

    time = datetime.now().strftime("%Y-%m-%d_%H:%M")
    log_dir = os.path.join(
        args.logdir, args.env, f'{args.algo}-seed{args.seed}-{time}')
    print("LOGDIR: ", log_dir)

    STATE_SHAPE = env.observation_space.shape
    ACTION_SHAPE = env.action_space.shape

    if args.algo == "SAC":
        trainer_class = ModelFreeTrainer
        algo = SAC(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            tune_alpha=args.tune_alpha,
            device=args.device,
            seed=args.seed,
            wandb=wandb
        )
    elif args.algo == "TD3":
        trainer_class = ModelFreeTrainer
        algo = TD3(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            device=args.device,
            expl_noise=args.expl_noise,
            seed=args.seed,
            wandb=wandb
        )
    elif args.algo == "DDPG":
        trainer_class = ModelFreeTrainer
        algo = DDPG(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            device=args.device,
            seed=args.seed,
            wandb=wandb
        )
    elif args.algo == "MOCCO":
        trainer_class = GemboTrainer
        algo = MOCCO(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            device=args.device,
            seed=args.seed,
            wandb=wandb
        )

    trainer = trainer_class(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        env=env,
        env_test=env_test,
        algo=algo,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        device=args.device,
        save_buffer_every=args.save_buffer_every,
        visualize_every=args.visualize_every,
        estimate_q_every=args.estimate_q_every,
        stdout_log_every=args.stdout_log_every,
        seed=args.seed,
        log_dir=log_dir,
        wandb=wandb
    )

    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='point_mass-easy')
    p.add_argument('--algo', type=str, default='MOCCO')
    p.add_argument('--num_steps', type=int, default=int(1e6))
    p.add_argument('--tune_alpha', action="store_true")
    p.add_argument("--exp_name", type=str, default="default")
    p.add_argument("--eval_interval", type=int, default=int(2e3))
    p.add_argument("--save_buffer_every", type=int, default=0)
    p.add_argument("--stdout_log_every", type=int, default=int(100000))
    p.add_argument("--visualize_every", type=int, default=0)
    p.add_argument("--estimate_q_every", type=int, default=0)
    p.add_argument("--ge", action="store_true")
    p.add_argument("--expl_noise", type=float, default=0.1)
    p.add_argument('--device', type=str, default="cuda:0")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument("--logdir", type=str, default="logs")
    args = p.parse_args()
    run(args)
