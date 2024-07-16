from typing import Tuple, Dict, Optional
from os import PathLike
from collections import OrderedDict

from omegaconf import OmegaConf, DictConfig
import gym
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as tf

from croco.models.croco import CroCoNet
from croco.models.croco_downstream import CroCoDownstreamBinocular

from habitat_baselines.rl.ppo.policy import NetPolicy, Net
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy.resnet import resnet18
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.common.baseline_registry import baseline_registry


class LinearHead(nn.Module):
    def __init__(self, proj_channels: int) -> None:
        super().__init__()
        self.proj_channels = proj_channels

    def setup(self, croco_net: CroCoNet) -> None:
        self.n_cls_token = getattr(croco_net, "n_cls_token", 0)
        self.proj = nn.Linear(croco_net.dec_embed_dim, self.proj_channels)

    def forward(self, dec_out: Tensor, *args, **kwargs) -> Tensor:
        return self.proj(dec_out[:, self.n_cls_token:, :]).flatten(1, -1)


class DEBiTBinocEncoder(CroCoDownstreamBinocular):
    def __init__(self, config: DictConfig) -> None:
        self.pre = tf.Compose(
            [
                tf.Resize(config.croco.img_size, antialias=True),
                tf.ConvertImageDtype(torch.float),
                tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        num_patches, rem = divmod(config.croco.img_size, config.croco.patch_size)
        assert rem == 0, "Patch size must divide image size"
        proj_channels, rem = divmod(config.binoc_repr_size, num_patches**2)
        assert rem == 0, "Binoc. repr. size must be a multiple of #patches"
        self.output_size = config.binoc_repr_size
        super().__init__(
            LinearHead(proj_channels),
            **OmegaConf.to_container(config.croco),
        )
        if config.pretrained_binoc_weights is not None:
            ckpt = torch.load(config.pretrained_binoc_weights, map_location="cpu")
            for k in list(ckpt["model"]):
                if k.startswith("head"):
                    ckpt["model"].pop(k)
            miss, extra = self.load_state_dict(ckpt["model"], strict=False)
            assert all(k.startswith("head") for k in miss)
            assert not extra
            for k, prm in self.named_parameters():
                if k.startswith("head"):
                    continue
                prm.requires_grad = False

    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        return super().forward(
            self.pre(observations["rgb"].permute(0, 3, 1, 2)),
            self.pre(observations["imagegoal"].permute(0, 3, 1, 2)),
        )


class DEBiTNet(Net):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.Space,
        config: DictConfig,
    ) -> None:
        super().__init__()
        monoc_resnet = ResNetEncoder(
            gym.spaces.Dict({"rgb": observation_space["rgb"]}),
            baseplanes=32,
            ngroups=16,
            make_backbone=resnet18,
        )
        c_out, h_out, w_out = monoc_resnet.output_shape
        self.monocular_encoder = nn.Sequential(
            OrderedDict(
                {
                    "resnet": monoc_resnet,
                    "flat": nn.Flatten(),
                    "fc": nn.Linear(c_out * h_out * w_out, config.monoc_repr_size),
                    "relu": nn.ReLU(True),
                }
            )
        )
        self.binocular_encoder = DEBiTBinocEncoder(config)
        self.prev_action_embedding = nn.Embedding(
            action_space.n + 1, config.act_repr_size
        )
        obs_repr_size = (
            config.monoc_repr_size + config.binoc_repr_size + config.act_repr_size
        )
        self.state_encoder = build_rnn_state_encoder(
            obs_repr_size, config.hidden_size, "GRU", config.num_recurrent_layers
        )

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_hidden_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
        rnn_build_seq_info: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        aux_s = {}
        monoc_repr = self.monocular_encoder({"rgb": observations["rgb"]})
        binoc_repr = self.binocular_encoder(observations)
        act_repr = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions.squeeze(-1) + 1, 0)
        )
        obs_repr = torch.cat((monoc_repr, binoc_repr, act_repr), -1)
        aux_s["perception_embed"] = obs_repr
        s_repr, hid = self.state_encoder(
            obs_repr, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_s["rnn_output"] = s_repr
        return s_repr, hid, aux_s

    @property
    def output_size(self) -> int:
        return self.state_encoder.rnn.hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers

    @property
    def is_blind(self) -> bool:
        return False

    @property
    def perception_embedding_size(self) -> int:
        return self.state_encoder.rnn.input_size


@baseline_registry.register_policy
class DEBiTPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.Space,
        debit_config: DictConfig,
        policy_config: Optional[DictConfig] = None,
        aux_loss_config: Optional[DictConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            DEBiTNet(
                observation_space,
                action_space,
                debit_config,
            ),
            action_space,
            policy_config,
            aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        observation_space: gym.spaces.Dict,
        action_space: gym.Space,
        **kwargs,
    ) -> "DEBiTPolicy":
        return cls(
            observation_space,
            action_space,
            config.debit,
            config.habitat_baselines.rl.policy,
            config.habitat_baselines.rl.auxiliary_losses,
        )
