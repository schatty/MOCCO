import os
import argparse
from datetime import datetime

import wandb

from algos.sac import SAC
from algos.td3 import TD3
from trainers.mf_trainer import ModelFreeTrainer
from env import make_env


def run(args):
    config = dict(
        env = args.env,
        num_steps = args.num_steps,
        algo = args.algo,
        device=args.device,
        seed = args.seed,
    )
    wandb.init(project="AAAI-baselines", config=config, group=f"{args.env}-{args.algo}-{args.exp_name}")

    env = make_env(args.env, args.seed)
    env_test = make_env(args.env, args.seed)

    time = datetime.now().strftime("%Y-%m-%d_%H:%M")
    log_dir = os.path.join(
        args.logdir, args.env, f'{args.algo}-seed{args.seed}-{time}')
    print("LOGDIR: ", log_dir)

    STATE_SHAPE = env.observation_space.shape
    ACTION_SHAPE = env.action_space.shape

    if args.algo == "SAC":
        algo = SAC(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            target_update_coef=args.tau,
            gamma=args.gamma,
            alpha_init=args.alpha_init,
            lr_actor=args.lr_alpha,
            lr_critic=args.lr_critic,
            lr_alpha=args.lr_alpha, 
            tune_alpha=args.tune_alpha,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            wandb=wandb
        )
    elif args.algo == "TD3":
        algo = TD3(
            state_shape=STATE_SHAPE,
            action_shape=ACTION_SHAPE,
            target_update_coef=args.tau,
            gamma=args.gamma,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            wandb=wandb
        )

    trainer = ModelFreeTrainer(
        state_shape=STATE_SHAPE,
        action_shape=ACTION_SHAPE,
        env=env,
        env_test=env_test,
        algo=algo,
        num_steps=args.num_steps,
        start_steps=args.start_steps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        device=args.device,
        log_dir=log_dir,
        seed=args.seed,
        save_buffer_every=args.save_buffer_every,
        visualize_every=args.visualize_every,
        estimate_q_every=args.estimate_q_every,
        stdout_log_every=args.stdout_log_every,
        wandb=wandb
    )

    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='HalfCheetah-v3')
    p.add_argument('--algo', type=str, default='SAC')
    p.add_argument('--num_steps', type=int, default=int(1e6))
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=5e-3)
    p.add_argument('--lr_actor', type=float, default=3e-4)
    p.add_argument('--lr_critic', type=float, default=1e-3)
    p.add_argument('--lr_alpha', type=float, default=1e-3)
    p.add_argument('--alpha_init', type=float, default=0.2)
    p.add_argument('--tune_alpha', action="store_true")
    p.add_argument('--buffer_size', type=int, default=int(3e6))
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--device', type=str, default="cuda:0")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--start_steps', type=int, default=int(10e3))
    p.add_argument("--logdir", type=str, default="logs")
    p.add_argument("--exp_name", type=str, default="default")
    p.add_argument("--eval_interval", type=int, default=int(2e3))
    p.add_argument("--save_buffer_every", type=int, default=0)
    p.add_argument("--stdout_log_every", type=int, default=int(100000))
    p.add_argument("--visualize_every", type=int, default=0)
    p.add_argument("--estimate_q_every", type=int, default=0)
    args = p.parse_args()
    run(args)
