import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import FQFAgent

import torch_xla.distributed.xla_multiprocessing as xmp

import reverb

def run(index, flags):
    with open(flags["config"]) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(flags["env_id"])
    test_env = make_pytorch_env(
        flags["env_id"], episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = flags["config"].split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', flags["env_id"], f'{name}-seed{flags["seed"]}-{time}')

    # Create the agent and run.
    agent = FQFAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=flags["seed"],
        cuda=flags["cuda"], **config)
    print(f"core {index} agent created, running now")



    agent.save_models(os.path.join('test_stonks'))
    print(f"{agent.device}: saved model")
    xm.rendezvous("saved") # note that the non-master nodes won't save 

    agent.load_models(os.path.join('test_stonks'))


    # if flags["start_from_checkpoint"]:
    #     STARTING_CHECKPOINT_DIR = os.path.join("starting_checkpoint")
    #     agent.load_models(STARTING_CHECKPOINT_DIR)
    #     print("checkpoint loaded successfully!")

def map_fn(index, flags, *args):
    print("flags:", flags)
    print("extra args:", args)
    run(index, flags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nprocs', type=int, default=1)
    parser.add_argument('--start_from_checkpoint', action='store_true')
    args = parser.parse_args()
    # wrapper(args)
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

    flags = vars(args)

    # keep the comma after flags, otherwise it tries to expand the dict
    xmp.spawn(map_fn, args=(flags,), nprocs=args.nprocs, start_method='spawn') # fork
