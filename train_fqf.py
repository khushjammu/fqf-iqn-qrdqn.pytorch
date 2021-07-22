import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import FQFAgent

import torch_xla.distributed.xla_multiprocessing as xmp

import reverb

def run(index, args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent and run.
    agent = FQFAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    print(f"core {index} agent created, running now")
    agent.run()

# def wrapper(args):
#     def map_fn(index):
#         run(index, args)


#     # start reverb server first here? then each client access it
#     # TODO: populate signature dynamically
    # replay_table = reverb.Table(
    #      name='replay_table',
    #      sampler=reverb.selectors.Uniform(),
    #      remover=reverb.selectors.Fifo(),
    #      max_size=10**6,
    #      rate_limiter=reverb.rate_limiters.MinSize(1),
    # )

    # reverb_server = reverb.Server([replay_table], port=8000)
    # print(reverb.Client('localhost:8000').server_info())
    
#     xmp.spawn(map_fn, args=(), nprocs=args.nprocs, start_method='fork')

def map_fn(index, args):
    run(index, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nprocs', type=int, default=1)
    args = parser.parse_args()
    wrapper(args)
    # run(args)

    replay_table = reverb.Table(
         name='replay_table',
         sampler=reverb.selectors.Uniform(),
         remover=reverb.selectors.Fifo(),
         max_size=10**6,
         rate_limiter=reverb.rate_limiters.MinSize(1),
    )

    reverb_server = reverb.Server([replay_table], port=8000)
    print(reverb.Client('localhost:8000').server_info())

    xmp.spawn(map_fn, args=(), nprocs=args.nprocs, start_method='spawn') # fork
