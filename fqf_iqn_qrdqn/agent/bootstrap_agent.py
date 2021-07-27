import torch
from torch.optim import Adam, RMSprop
import numpy as np

from fqf_iqn_qrdqn.model import EnsembleNet
from fqf_iqn_qrdqn.utils import calculate_quantile_huber_loss, disable_gradients, evaluate_quantile_at_action, update_params

from .base_agent import BaseAgent


class BootstrapAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=32, k=9, num_cosines=64, ent_coef=0,
                 kappa=1.0, lr=5e-5,
                 memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0):
        super(BootstrapAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, dueling_net, noisy_net, use_per, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # Online network.
        self.online_net = EnsembleNet(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, k=k,
            num_cosines=num_cosines, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)

        # Target network.
        self.target_net = EnsembleNet(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, k=k,
            num_cosines=num_cosines, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.N = N
        self.kappa = kappa

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2/batch_size)

    def update_target(self):
        self.target_net.dqn_net.load_state_dict(
            self.online_net.dqn_net.state_dict())
        self.target_net.quantile_net.load_state_dict(
            self.online_net.quantile_net.state_dict())
        self.target_net.cosine_net.load_state_dict(
            self.online_net.cosine_net.state_dict())

    def learn(self):
        ### TODO: MASKING
        self.learning_steps += 1
        self.online_net.sample_noise()
        self.target_net.sample_noise()


        # fetch sample from  reverb
        samples = list(self.memory.sample('replay_table', num_samples=self.batch_size))

        # stack individual samples into batch
        states = np.stack([sample[0].data[0] for sample in samples], axis=0)
        actions = np.stack([sample[0].data[1] for sample in samples], axis=0)
        rewards = np.stack([sample[0].data[2] for sample in samples], axis=0)
        next_states = np.stack([sample[0].data[3] for sample in samples], axis=0)
        dones = np.stack([sample[0].data[4] for sample in samples], axis=0)
        masks = np.stack([sample[0].data[5] for sample in samples], axis=0)

        # convert everything for torch
        states = torch.ByteTensor(states).to(self.device).float() / 255.
        next_states = torch.ByteTensor(
            next_states).to(self.device).float() / 255.
        actions = torch.LongTensor(actions).to(self.device)
        actions = torch.reshape(actions, (self.batch_size, 1))

        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = torch.reshape(rewards, (self.batch_size, 1))
        
        dones = torch.FloatTensor(dones).to(self.device)
        dones = torch.reshape(dones, (self.batch_size, 1))

        masks = torch.FloatTensor(masks).to(self.device)
        masks = torch.reshape(masks, (self.batch_size, 1))

        weights = None

        # print("states shape:", states, states.shape, type(states), states.type)
        # print("next_states shape:", next_states, next_states.shape, type(next_states), next_states.type)
        # print("action shape:", actions, actions.shape, type(actions), actions.type)
        # print("reward shape:", rewards, rewards.shape, type(rewards), rewards.type)
        # print("done shape:", dones, dones.shape, type(dones), dones.type)

        # TODO: implement prioritised replay
        # if self.use_per:
        #     (states, actions, rewards, next_states, dones), weights =\
        #         self.memory.sample(self.batch_size)
        # else:
        #     states, actions, rewards, next_states, dones =\
        #         self.memory.sample(self.batch_size)
        #     weights = None

        # Calculate embeddings of current states.
        state_embeddings = self.online_net.calculate_state_embeddings(states)

        all_target_next_Qs = [n.detach() for n in self.target_net.calculate_q(torch.Tensor(nexts), state_embeddings=None)]
        all_Qs = self.online_network.calculate_q(states=torch.Tensor(inputs), state_embeddings=None)

        self.optim.zero_grad()

        for k in range(self.k):
            next_Qs = all_target_next_Qs[k]
            next_max_Qs = next_Qs.max(1)[0]
            next_max_Qs = next_max_Qs.squeeze()

            # mask based on if it is end of episode or not
            next_max_Qs = (1.0 - torch.Tensor(dones)) * next_max_Qs
            target_Qs = torch.Tensor(np.array(rewards).astype("float32")) + self.gamma * next_max_Qs

            # get current step predictions
            Qs = all_Qs[k]
            Qs = Qs.gather(1, torch.LongTensor(np.array(actions)[:, None].astype("int32")))
            Qs = Qs.squeeze()

            # BROADCASTING! NEED TO MAKE SURE DIMS MATCH
            # need to do updates on each head based on experience mask
            full_loss = (Qs - target_Qs) ** 2
            full_loss = mask[:, k] * full_loss
            loss = torch.mean(full_loss)

            loss.backward(retain_graph=True)
            for param in self.online_network.parameters():
                if param.grad is not None:
                    # Multiply grads by 1 / K
                    param.grad.data *= 1. / N_ENSEMBLE
            epoch_losses[k] += loss.detach().cpu().numpy()
            epoch_steps[k] += 1.

        # TODO: gradient clipping
        # torch.nn.utils.clip_grad_value_(self.online_network.parameters(), CLIP_GRAD)

        self.optim.step()

        # # Calculate fractions of current states and entropies.
        # taus, tau_hats, entropies =\
        #     self.online_net.calculate_fractions(
        #         state_embeddings=state_embeddings.detach())

        # # Calculate quantile values of current states and actions at tau_hats.
        # current_sa_quantile_hats = evaluate_quantile_at_action(
        #     self.online_net.calculate_quantiles(
        #         tau_hats, state_embeddings=state_embeddings),
        #     actions)
        # assert current_sa_quantile_hats.shape == (
        #     self.batch_size, self.N, 1)

        # # NOTE: Detach state_embeddings not to update convolution layers. Also,
        # # detach current_sa_quantile_hats because I calculate gradients of taus
        # # explicitly, not by backpropagation.
        # fraction_loss = self.calculate_fraction_loss(
        #     state_embeddings.detach(), current_sa_quantile_hats.detach(),
        #     taus, actions, weights)

        # quantile_loss, mean_q, errors = self.calculate_quantile_loss(
        #     state_embeddings, tau_hats, current_sa_quantile_hats, actions,
        #     rewards, next_states, dones, weights)
        # assert errors.shape == (self.batch_size, 1)

        # entropy_loss = -self.ent_coef * entropies.mean()

        # update_params(
        #     self.fraction_optim, fraction_loss + entropy_loss,
        #     networks=[self.online_net.fraction_net], retain_graph=True,
        #     grad_cliping=self.grad_cliping)
        # update_params(
        #     self.quantile_optim, quantile_loss,
        #     networks=[
        #         self.online_net.dqn_net, self.online_net.cosine_net,
        #         self.online_net.quantile_net],
        #     retain_graph=False, grad_cliping=self.grad_cliping)

        # TODO: implement prioritised replay
        # if self.use_per:
            # self.memory.update_priority(errors)

        # self.writer should be None for non-master-ordinal cores
        if self.learning_steps % self.log_interval == 0 and self.writer:
            self.writer.add_scalar(
                'loss/stonks', loss.detach().item(), 4*self.steps)
            # self.writer.add_scalar(
            #     'loss/fraction_loss', fraction_loss.detach().item(),
            #     4*self.steps)
            # self.writer.add_scalar(
            #     'loss/quantile_loss', quantile_loss.detach().item(),
            #     4*self.steps)
            # if self.ent_coef > 0.0:
            #     self.writer.add_scalar(
            #         'loss/entropy_loss', entropy_loss.detach().item(),
            #         4*self.steps)

            # self.writer.add_scalar('stats/mean_Q', mean_q, 4*self.steps)
            # self.writer.add_scalar(
            #     'stats/mean_entropy_of_value_distribution',
            #     entropies.mean().detach().item(), 4*self.steps)

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()
        episode_k = np.random.choice(range(self.k))



        def _insert_to_reverb(state, action, reward, next_state, done, mask):
            state_mod = np.empty(self.env.observation_space.shape, dtype=np.uint8)
            state_mod[...] = state

            next_state_mod = np.empty(self.env.observation_space.shape, dtype=np.uint8)
            next_state_mod[...] = next_state

            self.memory.insert([state_mod, [action], [reward], next_state_mod, [done], mask], priorities={"replay_table": 1.0})
            # self.memory.insert([state, action, reward, next_state, done])

        while (not done) and episode_steps <= self.max_episode_steps:
            # NOTE: Noises can be sampled only after self.learn(). However, I
            # sample noises before every action, which seems to lead better
            # performances.
            self.online_net.sample_noise()

            if self.is_random(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state, episode_k)

            next_state, reward, done, _ = self.env.step(action)

            mask = torch.poisson(torch.tensor(1.))

            # To calculate efficiently, I just set priority=max_priority here.
            # self.memory.append(state, action, reward, next_state, done)
            # client.insert([0, 1], priorities={'my_table': 1.0})

            # torch.Size([32, 4, 84, 84]) torch.Size([32, 1]) torch.Size([32, 1]) torch.Size([32, 4, 84, 84]) torch.Size([32, 1])

            _insert_to_reverb(state, action, reward, next_state, done, mask)

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

    # def calculate_fraction_loss(self, state_embeddings, sa_quantile_hats, taus,
    #                             actions, weights):
    #     assert not state_embeddings.requires_grad
    #     assert not sa_quantile_hats.requires_grad

    #     batch_size = state_embeddings.shape[0]

    #     with torch.no_grad():
    #         sa_quantiles = evaluate_quantile_at_action(
    #             self.online_net.calculate_quantiles(
    #                 taus=taus[:, 1:-1], state_embeddings=state_embeddings),
    #             actions)
    #         assert sa_quantiles.shape == (batch_size, self.N-1, 1)

    #     # NOTE: Proposition 1 in the paper requires F^{-1} is non-decreasing.
    #     # I relax this requirements and calculate gradients of taus even when
    #     # F^{-1} is not non-decreasing.

    #     values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
    #     signs_1 = sa_quantiles > torch.cat([
    #         sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
    #     assert values_1.shape == signs_1.shape

    #     values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
    #     signs_2 = sa_quantiles < torch.cat([
    #         sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1)
    #     assert values_2.shape == signs_2.shape

    #     gradient_of_taus = (
    #         torch.where(signs_1, values_1, -values_1)
    #         + torch.where(signs_2, values_2, -values_2)
    #     ).view(batch_size, self.N-1)
    #     assert not gradient_of_taus.requires_grad
    #     assert gradient_of_taus.shape == taus[:, 1:-1].shape

    #     # Gradients of the network parameters and corresponding loss
    #     # are calculated using chain rule.
    #     if weights is not None:
    #         fraction_loss = ((
    #             (gradient_of_taus * taus[:, 1:-1]).sum(dim=1, keepdim=True)
    #         ) * weights).mean()
    #     else:
    #         fraction_loss = \
    #             (gradient_of_taus * taus[:, 1:-1]).sum(dim=1).mean()

    #     return fraction_loss

    # def calculate_quantile_loss(self, state_embeddings, tau_hats,
    #                             current_sa_quantile_hats, actions, rewards,
    #                             next_states, dones, weights):
    #     assert not tau_hats.requires_grad

    #     with torch.no_grad():
    #         # NOTE: Current and target quantiles share the same proposed
    #         # fractions to reduce computations. (i.e. next_tau_hats = tau_hats)

    #         # Calculate Q values of next states.
    #         if self.double_q_learning:
    #             # Sample the noise of online network to decorrelate between
    #             # the action selection and the quantile calculation.
    #             self.online_net.sample_noise()
    #             next_q = self.online_net.calculate_q(states=next_states)
    #         else:
    #             next_state_embeddings =\
    #                 self.target_net.calculate_state_embeddings(next_states)
    #             next_q = self.target_net.calculate_q(
    #                 state_embeddings=next_state_embeddings,
    #                 fraction_net=self.online_net.fraction_net)

    #         # Calculate greedy actions.
    #         next_actions = torch.argmax(next_q, dim=1, keepdim=True)
    #         assert next_actions.shape == (self.batch_size, 1)

    #         # Calculate features of next states.
    #         if self.double_q_learning:
    #             next_state_embeddings =\
    #                 self.target_net.calculate_state_embeddings(next_states)

    #         # Calculate quantile values of next states and actions at tau_hats.
    #         next_sa_quantile_hats = evaluate_quantile_at_action(
    #             self.target_net.calculate_quantiles(
    #                 taus=tau_hats, state_embeddings=next_state_embeddings),
    #             next_actions).transpose(1, 2)
    #         assert next_sa_quantile_hats.shape == (
    #             self.batch_size, 1, self.N)

    #         # Calculate target quantile values.
    #         target_sa_quantile_hats = rewards[..., None] + (
    #             1.0 - dones[..., None]) * self.gamma_n * next_sa_quantile_hats
    #         assert target_sa_quantile_hats.shape == (
    #             self.batch_size, 1, self.N)

    #     td_errors = target_sa_quantile_hats - current_sa_quantile_hats
    #     assert td_errors.shape == (self.batch_size, self.N, self.N)

    #     quantile_huber_loss = calculate_quantile_huber_loss(
    #         td_errors, tau_hats, weights, self.kappa)

    #     return quantile_huber_loss, next_q.detach().mean().item(), \
    #         td_errors.detach().abs().sum(dim=1).mean(dim=1, keepdim=True)
