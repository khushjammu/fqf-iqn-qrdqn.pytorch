from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import torch_xla
import torch_xla.core.xla_model as xm

import reverb

# from fqf_iqn_qrdqn.memory import LazyMultiStepMemory, \
#     LazyPrioritizedMultiStepMemory
from fqf_iqn_qrdqn.utils import RunningMeanStats, LinearAnneaer


class BaseAgent(ABC):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=5.0, cuda=True, seed=0):

        self.env = env
        self.test_env = test_env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = xm.xla_device()
        # self.device = torch.device(
        #     "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.online_net = None
        self.target_net = None

        # Replay memory which is memory-efficient to store stacked frames.
        print("use_per ignored, need to implement prioritised replay")

        # reverb client to sample from the reverb server
        self.memory = reverb.Client('localhost:8000')


        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)


        if xm.is_master_ordinal():
            self.writer = SummaryWriter(log_dir=self.summary_dir)
        else:
            self.writer = None

        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.double_q_learning = double_q_learning
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.use_per = use_per

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(
            1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def is_random(self, eval=False):
        # Use e-greedy for evaluation.
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        if self.noisy_net:
            return False
        return np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict())

    def explore(self):
        # Act with randomness.
        action = self.env.action_space.sample()
        return action

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.online_net.calculate_q(states=state).argmax().item()
        return action

    @abstractmethod
    def learn(self):
        pass

    def save_models(self, save_dir):
        uuid = str(self.steps)
        if xm.is_master_ordinal() and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # we need to make the filenames unique b/c xm.save doesn't overwrite by default
        xm.save(
            self.online_net.state_dict(),
            os.path.join(save_dir, f'{uuid}-online_net.pth'))
        xm.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, f'{uuid}-target_net.pth'))

    def load_models(self, save_dir):
        self.online_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'online_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        def _insert_to_reverb(state, action, reward, next_state, done):
            state_mod = np.empty(self.env.observation_space.shape, dtype=np.uint8)
            state_mod[...] = state

            next_state_mod = np.empty(self.env.observation_space.shape, dtype=np.uint8)
            next_state_mod[...] = next_state

            self.memory.insert([state_mod, [action], [reward], next_state_mod, [done]], priorities={"replay_table": 1.0})
            # self.memory.insert([state, action, reward, next_state, done])

        while (not done) and episode_steps <= self.max_episode_steps:
            # NOTE: Noises can be sampled only after self.learn(). However, I
            # sample noises before every action, which seems to lead better
            # performances.
            self.online_net.sample_noise()

            if self.is_random(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)

            # To calculate efficiently, I just set priority=max_priority here.
            # self.memory.append(state, action, reward, next_state, done)
            # client.insert([0, 1], priorities={'my_table': 1.0})

            # torch.Size([32, 4, 84, 84]) torch.Size([32, 1]) torch.Size([32, 1]) torch.Size([32, 4, 84, 84]) torch.Size([32, 1])

            _insert_to_reverb(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()

        # We log running mean of stats.
        self.train_return.append(episode_return)

        # We log evaluation results along with training frames = 4 * steps.
        if self.episodes % self.log_interval == 0 and self.writer:
            self.writer.add_scalar(
                'return/train', self.train_return.get(), 4 * self.steps)

        # print
        xm.master_print(f'Episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'return: {episode_return:<5.1f}')

    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.is_update():
            self.learn()

        if self.steps % self.eval_interval == 0:
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'final'))
            self.online_net.train()

    def evaluate(self):
        self.online_net.eval()
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                if self.is_random(eval=True):
                    action = self.explore()
                else:
                    action = self.exploit(state)

                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        # We log evaluation results along with training frames = 4 * steps.
        if self.writer:
            self.writer.add_scalar(
                'return/test', mean_return, 4 * self.steps)
        xm.master_print('-' * 60)
        xm.master_print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        xm.master_print('-' * 60)

    def __del__(self):
        self.env.close()
        self.test_env.close()

        if self.writer:
            self.writer.close()
