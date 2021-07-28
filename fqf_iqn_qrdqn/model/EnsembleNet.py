from torch import nn

from .base_model import BaseModel
from fqf_iqn_qrdqn.network import DQNBase, NoisyLinear

class HeadNet(nn.Module):
    def __init__(self, embedding_dim, num_actions):
        super(HeadNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions), # can think about predicting distribution of Qs
        )

    def forward(self, x):
        return self.network(x)

class EnsembleNet(BaseModel):

    def __init__(self, num_channels, num_actions, N=200, k=9, embedding_dim=7*7*64,
                 dueling_net=False, noisy_net=False):
        super(EnsembleNet, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels)
        # Quantile network.
        if not dueling_net:
            pass
            # self.head_net = nn.Sequential(
            #     linear(embedding_dim, 512),
            #     nn.ReLU(),
            #     linear(512, num_actions), # can think about predicting distribution of Qs
            # )
        else:
            raise NotImplementedError()

            # self.advantage_net = nn.Sequential(
            #     linear(embedding_dim, 512),
            #     nn.ReLU(),
            #     linear(512, num_actions * N),
            # )
            # self.baseline_net = nn.Sequential(
            #     linear(embedding_dim, 512),
            #     nn.ReLU(),
            #     linear(512, N),
            # )

        self.net_list = nn.ModuleList([HeadNet(embedding_dim, num_actions) for k in range(k)])

        self.N = N
        self.K = k
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def forward(self, k, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        if not self.dueling_net:
            qs = self.net_list[k](state_embeddings)
            # quantiles = self.q_net(
            #     state_embeddings).view(batch_size, self.N, self.num_actions)

        else:
            raise NotImplementedError()
            # advantages = self.advantage_net(
            #     state_embeddings).view(batch_size, self.N, self.num_actions)
            # baselines = self.baseline_net(
            #     state_embeddings).view(batch_size, self.N, 1)
            # quantiles = baselines + advantages\
                # - advantages.mean(dim=2, keepdim=True)

        assert qs.shape == (batch_size, self.num_actions)

        return qs

    def calculate_q(self, k=None, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        # Calculate quantiles.
        if k is not None:
            q = self(k, states=states, state_embeddings=state_embeddings)
        else:
            qs = [self(i, states=states, state_embeddings=state_embeddings) for i in range(self.K)]
            return qs




        # Calculate expectations of value distributions.
        # q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q

    ### TODO: IMPLEMENT QUANTILES AND DUELING
