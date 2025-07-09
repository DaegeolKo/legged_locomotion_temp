import torch
import torch.nn as nn
# from IPython.terminal.shortcuts.auto_suggest import accept_token
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic
from torch.cuda.amp import autocast
import math
from typing import Optional
# from performer_pytorch import Performer
# from bitnet import BitNetTransformer
# from bitnet.bit_transformer import Transformer as BitNetTransformer
# from bitnet import BitMGQA, BitFeedForward


def generate_square_subsequent_mask(size, device):
    """
    Generate a square mask for the sequence. The mask will prevent attention to future positions.
    """
    if size <= 0:
        raise ValueError(f"Mask size must be greater than 0, got {size}")
    return torch.triu(torch.ones(size, size, device=device), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # # self.pe = pe
        # self.register_buffer('pe', pe)

        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        # x: (batch, sequence, dim)
        seq_len = x.size(1)  # sequence length
        pe = self.pe[:seq_len, :].flip(0).unsqueeze(0)  # (1, sequence, dim)
        return x + pe.to(x.device)

class HistoryBuffer(nn.Module):
    def __init__(self, history_buffer_length : int, device):
        super().__init__()
        self.history_buffer_length = history_buffer_length
        self.device = device
        self.history_buffer = torch.empty(0, self.history_buffer_length, 0, device=self.device) # (batch, sequence, dim_obs)
        self.history_buffer_mask = torch.empty(0, self.history_buffer_length, dtype=torch.bool, device=self.device)
        # self.prev_dones = torch.empty(0, dtype=torch.bool, device=self.device) # (batch,)
        self.obs = torch.empty(0, 0, device=self.device) # (batch, dim_obs)

    def initialize_history(self, initial_obs, history_buffer_length : int =-1):
        """Initialize the observation history buffer."""
        if history_buffer_length > 0:
            self.history_buffer_length = history_buffer_length
        if initial_obs.dim() == 1:
            initial_obs = initial_obs.unsqueeze(-1)
        self.history_buffer = initial_obs.unsqueeze(1).repeat(1, self.history_buffer_length, 1) # (batch, sequence, dim_obs)
        self.history_buffer[:, :-1, 0 : 3+3] = 0.0 # linvel & angvel
        self.history_buffer[:, :-1, 3+3+3 : 3+3+3+3] = 0.0 # commands
        self.history_buffer_mask = torch.ones([initial_obs.shape[0], self.history_buffer_length], dtype=torch.bool, device=self.device)
        self.history_buffer_mask[:,-1] = False

    def reset_buffer_with_dones(self, dones, obs):
        needs_reset = dones.to(torch.bool)
        history_buffer_temp = self.history_buffer.clone()
        history_buffer_temp[needs_reset, :, :] = obs[needs_reset].unsqueeze(1).repeat(1, self.history_buffer.size(1), 1)
        history_buffer_temp[needs_reset, :, 0 : 3+3] = 0.0 # linvel & angvel
        history_buffer_temp[needs_reset, :, 3+3+3 : 3+3+3+3] = 0.0 # commands
        self.history_buffer = history_buffer_temp

        history_buffer_mask_temp = self.history_buffer_mask.clone()
        history_buffer_mask_temp[needs_reset, :] = True
        # history_buffer_mask_temp[needs_reset, -1] = False
        self.history_buffer_mask = history_buffer_mask_temp

    def update_buffer(self, obs, dones: Optional[torch.Tensor] = None):
        # always save obs for manual reset
        self.obs = obs
        # # init dones in case of None
        # if dones is None:
        #     dones = torch.empty(0, dtype=torch.bool, device=self.device)
        """"Update the Transformer buffer based on dones."""
        # for the very first run:
        if self.history_buffer.size(1) == 0 or self.history_buffer.size(0) == 0 or self.history_buffer.size(0) != obs.size(0) or self.history_buffer.size(-1) != obs.size(-1):
            # if history_buffer sequence == 0 or batch size == 0 or batch size is different from obs batch size or dim is different:
            self.initialize_history(obs)
        else:
            if dones is not None: # done before. need to reset current with current obs. if no dones, pass
                self.reset(dones)
            self.history_buffer = torch.cat([self.history_buffer[:, 1:, :], obs.unsqueeze(1)], dim=1)  # (batch, sequence, dim_obs)
            self.history_buffer_mask = torch.cat(
                [self.history_buffer_mask[:, 1:],
                 torch.zeros((self.history_buffer_mask.size(0), 1), dtype=torch.bool, device=self.device)],
                dim=1
            )
            #
            # # if dones is not None:  # if dones is None, it means inference, which does not have "done" (or end)
            # if self.prev_dones.size(0) != 0 and self.prev_dones.any() == True: # if prev_dones is not None, need to reset
            #     # needs_reset = self.prev_dones.bool()
            #     needs_reset = self.prev_dones.to(torch.bool)
            #     '''
            #         RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
            #         You can make a clone to get a normal tensor before doing inplace update.
            #         See https://github.com/pytorch/rfcs/pull/17 for more details.
            #     '''
            #     history_buffer_temp = self.history_buffer.clone()
            #     history_buffer_temp[needs_reset, :, :] = obs[needs_reset].unsqueeze(1).repeat(
            #         1, self.history_buffer.size(1), 1) #(batch, sequence, n_dim)
            #     self.history_buffer = history_buffer_temp
            # self.prev_dones = dones


    def get(self, sampling_interval: int):
        if sampling_interval == 1:
            return self.history_buffer, self.history_buffer_mask
        elif sampling_interval > 1:
            return self.history_buffer[:, ::sampling_interval, :], self.history_buffer_mask[:, ::sampling_interval]
        else:
            raise ValueError(f"Unsupported sampling interval: {sampling_interval}")

    def reset(self, dones):
        # self.prev_dones = dones
        # manual reset when dones are given
        self.reset_buffer_with_dones(dones, self.obs)


    def forward(self, obs, dones: Optional[torch.Tensor] = None, sampling_interval: int = 1):
        self.update_buffer(obs, dones)
        return self.get(sampling_interval)

class TransformerModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=256,
        nhead=4,
        num_layers=3,
        history_buffer_length=20,
        future_step_size=1,
        sampling_interval=1,
        dropout=0.0,
        device="cuda:0",
        activation='relu',
        std=False,
        normalized_policy=False,
        # use_performer=False,  # Default to PyTorch Transformer
        **kwargs,
    ):
        super().__init__()
        # self.use_performer =use_performer

        # Model configuration
        self.d_model = d_model
        self.history_buffer_length = history_buffer_length
        self.future_step_size = future_step_size
        self.sampling_interval = sampling_interval
        self.device = device

        # Buffer initialization
        self.history_buffer = HistoryBuffer(history_buffer_length=history_buffer_length, device=self.device)

        # Layers
        self.obs_projector = nn.Linear(input_dim, d_model)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.activation_layer = get_activation(activation)
        self.positional_encoding = PositionalEncoding(d_model, max_len=1024)

        # Learnable dummy token for future steps
        self.dummy_token = nn.Parameter(torch.zeros(1, self.future_step_size, d_model, device=self.device)) # (batch, future_sequence_length, d_model)

        self.std = std
        self.normalized_policy = normalized_policy
        # networks
        # self.transformer = nn.Transformer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     num_encoder_layers=num_layers,
        #     dropout=dropout,
        #     activation=activation,
        #     # batch_first=True,
        # )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers  # Encoder Layer 수
        )

    def forward(self, obs, masks: Optional[torch.Tensor] = None):
        """Compute the forward pass for the Transformer."""
        # self.history_buffer.update_buffer(obs, masks)
        # sampled_history = self.history_buffer.get(self.sampling_interval)
        sampled_history, sampled_history_mask = self.history_buffer(obs, masks, self.sampling_interval)

        projected_history = self.obs_projector(sampled_history)

        # Prepare input for the Transformer
        history_with_future_token = torch.cat(
            [projected_history, self.dummy_token.repeat(projected_history.size(0), 1,  1)],
            dim=1
        )

        history_mask_for_future_token = torch.cat(
            [sampled_history_mask, torch.zeros([sampled_history_mask.size(0), 1], dtype=torch.bool, device=self.device)],
            dim=-1
        )

        history_with_positional_encoding = self.positional_encoding(history_with_future_token)
        # history_with_positional_encoding = self.activation_layer(history_with_positional_encoding)

        # history_with_positional_encoding = history_with_positional_encoding.permute(1,0,2)
        # transformer_output = self.transformer(src=history_with_positional_encoding, tgt=history_with_positional_encoding)
        # transformer_output = self.transformer(src=history_with_positional_encoding, src_key_padding_mask=history_mask_for_future_token)
        transformer_output = self.transformer(src=history_with_positional_encoding)
        # transformer_output = transformer_output.permute(1,0,2)
        # transformer_output = self.activation_layer(transformer_output)

        output = self.output_layer(transformer_output[:, -1, :])

        upper_bound = 0.7
        if self.std:
            output_dim = output.size(-1)
            mean = output[:, :output_dim//2]
            logvar = output[:, output_dim//2:]
            if self.normalized_policy:
                mean = upper_bound * torch.tanh(mean)
            return mean, logvar
        else:
            if self.normalized_policy:
                output = upper_bound * torch.tanh(output)
            return output
        # output_dim = output.size(-1)
        # mean = output[:, :output_dim//2]
        # logvar = output[:, output_dim//2:]
        # if self.normalized_policy:
        #     mean = torch.tanh(mean)
        # return mean, logvar

    def reset(self, dones):
        self.history_buffer.reset(dones)

class ActorCriticTransformer(nn.Module):
    is_recurrent: int = -1

    def __init__(
            self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[256, 256, 256],
            activation="gelu",
            init_noise_std=1.0,
            history_buffer_length=3,
            sampling_interval=1,
            future_step_size=1,
            device="cuda:0",
            policy_std=False,
            value_std=False,
            normalized_policy=False,
            **kwargs,
    ):
        # activation = 'gelu'
        # # activation = 'relu'
        # super().__init__(
        #     num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation,
        #     init_noise_std, **kwargs
        # )
        super(ActorCriticTransformer, self).__init__()
        self.policy_std = policy_std
        self.value_std = value_std

        self.actor = TransformerModule(input_dim=num_actor_obs, output_dim=num_actions,
                                       d_model=256, nhead=4, num_layers=4,
                                       history_buffer_length=history_buffer_length, sampling_interval=sampling_interval,
                                       future_step_size=future_step_size, normalized_policy=normalized_policy, std=policy_std,
                                       activation=activation, device=device)
                                       # dropout=0.1)

        self.critic = TransformerModule(input_dim=num_critic_obs, output_dim=1,
                                        d_model=256, nhead=4, num_layers=6,
                                        history_buffer_length=history_buffer_length, sampling_interval=sampling_interval,
                                        future_step_size=future_step_size, std=value_std,
                                        activation=activation, device=device)
                                        # dropout=0.1)
        # Action noise
        self.logvar = nn.Parameter(torch.log(init_noise_std ** 2 * torch.ones(num_actions, device=device)))
        self.init_noise_std = init_noise_std
        # self.std = nn.Parameter(self.init_noise_std * torch.ones(num_actions, device=device))
        # self.std = init_noise_std * torch.ones(num_actions, device=device)
        # self.std = None
        self.distribution = None
        self.distribution_value = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @property
    def std(self):
        return torch.sqrt( torch.exp(self.logvar) )

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, masks=None):
        """Update the distribution using the predicted trajectory."""
        output = self.actor(observations, masks)
        if self.policy_std:
            mean, logvar = output
            self.std = self.init_noise_std * torch.sqrt(torch.exp(logvar))
        else:
            mean = output
        # mean = torch.clip(mean, min=-0.7, max=0.7)
        # self.std = 0.1 * torch.sqrt(torch.exp(logvar))
        self.distribution = Normal(mean, self.std)

    def act(self, observations, masks=None, hidden_states=None):
        """Act based on the current policy and update the buffer."""
        self.update_distribution(observations, masks)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """Deterministically output actions."""
        output = self.actor(observations)
        if self.policy_std:
            return output[0]
        else:
            return output

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate the value function."""
        return self.critic(critic_observations, masks)

    # def update_distribution_value(self, critic_observations, masks=None):
    #     mean, logvar = self.critic(critic_observations, masks)
    #     std = 0.01 * torch.sqrt(torch.exp(logvar))
    #     self.distribution_value = Normal(mean, std)
    #
    # def evaluate(self, critic_observations, masks=None, hidden_states=None):
    #     self.update_distribution_value(critic_observations, masks)
    #     return self.distribution_value.sample()

    def reset(self, dones=None):
        self.actor.reset(dones)
        self.critic.reset(dones)

    # def reset(self, dones=None):
    #     """Reset history buffer for environments that are done."""
    #     if dones is not None and self.history_buffer is not None:
    #         for idx, done in enumerate(dones):
    #             if done:
    #                 self.history_buffer[:, idx, :] = 0.0

# class PerformerModule(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         d_model=256,
#         nhead=4,
#         num_layers=3,
#         history_buffer_length=5,
#         future_step_size=1,
#         sampling_interval=1,
#         dropout=0.0,
#         device="cuda:0",
#         activation='relu',
#         # use_performer=False,  # Default to PyTorch Transformer
#         **kwargs,
#     ):
#         # activation = 'gelu'
#         super().__init__()
#         # self.use_performer =use_performer
#
#         # Model configuration
#         self.d_model = d_model
#         self.history_buffer_length = history_buffer_length
#         self.future_step_size = future_step_size
#         self.sampling_interval = sampling_interval
#         self.device = device
#
#         # Buffer initialization
#         self.history_buffer = HistoryBuffer(history_buffer_length=history_buffer_length, device=self.device)
#
#         # Layers
#         self.obs_projector = nn.Linear(input_dim, d_model)
#         self.positional_encoding = PositionalEncoding(d_model, max_len=512)
#         self.output_layer = nn.Linear(d_model, output_dim)
#
#         # Learnable dummy token
#         self.dummy_token = nn.Parameter(torch.zeros(1, self.future_step_size, d_model, device=self.device)) # (batch, future_sequence_length, d_model)
#
#         # networks
#         self.transformer = Performer(
#             dim=d_model,
#             depth=num_layers,
#             heads=nhead,
#             dim_head=d_model // nhead,
#             # ff_dropout=dropout,
#             # attn_dropout=dropout,
#             ff_mult=4,
#             kernel_fn=get_activation(activation),
#             causal=False,  # do not need Autoregressive
#         )
#
#     def forward(self, obs, masks: Optional[torch.Tensor] = None):
#         """Compute the forward pass for the Transformer."""
#         # self.history_buffer.update_buffer(obs, masks)
#         # sampled_history = self.history_buffer.get(self.sampling_interval)
#         sampled_history, sampled_history_mask = self.history_buffer(obs, masks, self.sampling_interval)
#
#         projected_history = self.obs_projector(sampled_history)
#
#         # Prepare input for the Transformer
#         history_with_future_token = torch.cat(
#             [projected_history, self.dummy_token.repeat(projected_history.size(0), 1,  1)],
#             dim=1
#         )
#
#         history_mask_for_future_token = torch.cat(
#             [sampled_history_mask, torch.zeros([sampled_history_mask.size(0), 1], dtype=torch.bool, device=self.device)],
#             dim=1
#         )
#
#         history_with_positional_encoding = self.positional_encoding(history_with_future_token)
#         # history_with_positional_encoding = self.activation_layer(history_with_positional_encoding)
#
#         # history_with_positional_encoding = history_with_positional_encoding.permute(1,0,2)
#         # transformer_output = self.transformer(src=history_with_positional_encoding, tgt=history_with_positional_encoding)
#         transformer_output = self.transformer(x=history_with_positional_encoding)
#         # transformer_output = self.transformer(x=history_with_positional_encoding, mask=history_mask_for_future_token)
#         # transformer_output = transformer_output.permute(1,0,2)
#         # transformer_output = self.activation_layer(transformer_output)
#
#         return self.output_layer(transformer_output[:, -1, :])
#
#     def reset(self, dones):
#         self.history_buffer.reset(dones)
#
# class ActorCriticPerformer(ActorCriticTransformer):
#     def __init__(
#         self,
#         num_actor_obs,
#         num_critic_obs,
#         num_actions,
#         actor_hidden_dims=[256, 256, 256],
#         critic_hidden_dims=[256, 256, 256],
#         activation="relu",
#         init_noise_std=1.0,
#         history_buffer_length=5,
#         sampling_interval=1,
#         future_step_size=1,
#         **kwargs,
#     ):
#         activation = 'gelu'
#         # ActorCriticTransformer 초기화
#         super().__init__(
#             num_actor_obs=num_actor_obs, num_actions=num_actions,
#             num_critic_obs=num_critic_obs,
#             actor_hidden_dims=actor_hidden_dims,
#             critic_hidden_dims=critic_hidden_dims,
#             init_noise_std=init_noise_std,
#             history_buffer_length=history_buffer_length, sampling_interval=sampling_interval,
#             future_step_size=future_step_size,
#             activation=activation,
#             **kwargs,
#         )
#
#         # Actor와 Critic에 Performer 설정
#         self.actor = PerformerModule(
#             input_dim=num_actor_obs, output_dim=num_actions,
#             d_model=256, nhead=4, num_layers=4,
#             history_buffer_length=history_buffer_length, sampling_interval=sampling_interval,
#             future_step_size=future_step_size,
#             activation=activation,
#             # dropout=0.1,
#         )
#
#         self.critic = PerformerModule(
#             input_dim=num_critic_obs, output_dim=1,
#             d_model=256, nhead=4, num_layers=6,
#             history_buffer_length=history_buffer_length, sampling_interval=sampling_interval,
#             future_step_size=future_step_size,
#             activation=activation,
#             # dropout=0.1,
#         )
#
#
# class BitNetTransformer(nn.Module):
#     """
#     Transformer module that applies multi-head attention and feed-forward layers.
#
#     Args:
#         dim (int): The dimension of the input and output tensors.
#         heads (int): The number of attention heads.
#         depth (int): The number of transformer layers.
#         ff_mult (int, optional): The multiplier for the hidden dimension in the feed-forward layers.
#             Defaults to 2.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.
#
#     Attributes:
#         layers (nn.ModuleList): List of multi-head attention layers.
#         ffn_layers (nn.ModuleList): List of feed-forward layers.
#
#     """
#
#     def __init__(
#             self, dim: int, heads: int, depth: int, ff_mult: int = 2, swish: bool = False, dropout: float = 0.0, *args, **kwargs
#     ):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.ffn_layers = nn.ModuleList([])
#
#         for _ in range(depth):
#             self.layers.append(BitMGQA(dim, heads, *args, **kwargs))
#
#             self.ffn_layers.append(
#                 BitFeedForward(
#                     dim,
#                     dim,
#                     ff_mult,
#                     swish=swish,
#                     post_act_ln=True,
#                     dropout=dropout,
#                 ),
#             )
#
#         # Norm
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         skip = x
#         for attn, ffn in zip(self.layers, self.ffn_layers):
#             x, _ = attn(x, x, x, is_causal=True, *args, **kwargs)
#             x = self.norm(x + skip)
#             x = ffn(x) + x
#         return x
#
# class BitNetTransformerModule(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         d_model=256,
#         nhead=4,
#         num_layers=3,
#         history_buffer_length=20,
#         future_step_size=1,
#         sampling_interval=1,
#         dropout=0.0,
#         device="cuda:0",
#         activation='relu',
#         # use_performer=False,  # Default to PyTorch Transformer
#         **kwargs,
#     ):
#         super().__init__()
#         # self.use_performer =use_performer
#
#         # Model configuration
#         self.d_model = d_model
#         self.history_buffer_length = history_buffer_length
#         self.future_step_size = future_step_size
#         self.sampling_interval = sampling_interval
#         self.device = device
#
#         # Buffer initialization
#         self.history_buffer = HistoryBuffer(history_buffer_length=history_buffer_length, device=self.device)
#
#         # Layers
#         self.obs_projector = nn.Linear(input_dim, d_model)
#         self.output_layer = nn.Linear(d_model, output_dim)
#         self.activation_layer = get_activation(activation)
#         self.positional_encoding = PositionalEncoding(d_model, max_len=512)
#
#         # Learnable dummy token
#         self.dummy_token = nn.Parameter(torch.zeros(1, self.future_step_size, d_model)) # (batch, future_sequence_length, d_model)
#
#         # networks
#         # self.transformer = nn.Transformer(
#         #     d_model=d_model,
#         #     nhead=nhead,
#         #     num_encoder_layers=num_layers,
#         #     dropout=dropout,
#         #     activation=activation,
#         #     # batch_first=True,
#         # )
#
#         self.transformer = BitNetTransformer(
#             dim=d_model,
#             heads=nhead,
#             depth=num_layers,
#             ff_mult=4,
#             swish=False,
#             dropout=dropout
#             # num_tokens=history_buffer_length + future_step_size,
#         )
#
#     def forward(self, obs, masks: Optional[torch.Tensor] = None):
#         """Compute the forward pass for the Transformer."""
#         # self.history_buffer.update_buffer(obs, masks)
#         # sampled_history = self.history_buffer.get(self.sampling_interval)
#         sampled_history, sampled_history_mask = self.history_buffer(obs, masks, self.sampling_interval)
#
#         projected_history = self.obs_projector(sampled_history)
#
#         # Prepare input for the Transformer
#         history_with_future_token = torch.cat(
#             [projected_history, self.dummy_token.repeat(projected_history.size(0), 1,  1)],
#             dim=1
#         )
#
#         history_mask_for_future_token = torch.cat(
#             [sampled_history_mask, torch.zeros([sampled_history_mask.size(0), 1], dtype=torch.bool, device=self.device)],
#             dim=-1
#         )
#
#         history_with_positional_encoding = self.positional_encoding(history_with_future_token)
#         # history_with_positional_encoding = self.activation_layer(history_with_positional_encoding)
#
#         # history_with_positional_encoding = history_with_positional_encoding.permute(1,0,2)
#         # transformer_output = self.transformer(src=history_with_positional_encoding, tgt=history_with_positional_encoding)
#         # transformer_output = self.transformer(src=history_with_positional_encoding, src_key_padding_mask=history_mask_for_future_token)
#         transformer_output = self.transformer(x=history_with_positional_encoding)
#         # transformer_output = transformer_output.permute(1,0,2)
#         # transformer_output = self.activation_layer(transformer_output)
#
#         return self.output_layer(transformer_output[:,-1, :])
#
#     def reset(self, dones):
#         self.history_buffer.reset(dones)
#
# class ActorCriticBitNetTransformer(ActorCriticTransformer):
#     def __init__(
#         self,
#         num_actor_obs,
#         num_critic_obs,
#         num_actions,
#         actor_hidden_dims=[256, 256, 256],
#         critic_hidden_dims=[256, 256, 256],
#         activation="relu",
#         init_noise_std=1.0,
#         history_buffer_length=3,
#         sampling_interval=1,
#         future_step_size=1,
#         **kwargs,
#     ):
#         # ActorCriticTransformer 초기화
#         super().__init__(
#             num_actor_obs=num_actor_obs, num_actions=num_actions,
#             num_critic_obs=num_critic_obs,
#             actor_hidden_dims=actor_hidden_dims,
#             critic_hidden_dims=critic_hidden_dims,
#             init_noise_std=init_noise_std,
#             history_buffer_length=history_buffer_length, sampling_interval=sampling_interval,
#             future_step_size=future_step_size,
#             activation=activation,
#             **kwargs,
#         )
#
#         # Actor와 Critic에 Performer 설정
#         self.actor = BitNetTransformerModule(
#             input_dim=num_actor_obs, output_dim=num_actions,
#             d_model=256, nhead=4, num_layers=4,
#             history_buffer_length=history_buffer_length, sampling_interval=sampling_interval,
#             future_step_size=future_step_size,
#             activation=activation,
#             # dropout=0.1,
#         )
#
#         self.critic = BitNetTransformerModule(
#             input_dim=num_critic_obs, output_dim=1,
#             d_model=256, nhead=4, num_layers=6,
#             history_buffer_length=history_buffer_length, sampling_interval=sampling_interval,
#             future_step_size=future_step_size,
#             activation=activation,
#             # dropout=0.1,
#         )




def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "gelu":
        return nn.GELU()
    else:
        print("invalid activation function!")
        return None
