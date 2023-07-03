from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from encoding import get_encoder
from plenoxels.models.density_fields import KPlaneDensityField
from plenoxels.models.kplane_field import KPlaneField
from plenoxels.ops.activations import init_density_activation
from plenoxels.raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)
from plenoxels.raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from plenoxels.utils.timer import CudaTimer

import torch.nn.functional as F
from torchvision.utils import save_image

class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.activation(out)

        return out
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=BasicBlock):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif l != num_layers - 1:
                net.append(block(self.dim_hidden, self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)

        return x


class LowrankModel(nn.Module):
    def __init__(self,
                 opt,
                 grid_config: Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 density_activation: Optional[str] = 'trunc_exp',
                 concat_features_across_scales: bool = False,
                 linear_decoder: bool = True,
                 linear_decoder_layers: Optional[int] = 1,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 # proposal-sampling arguments
                 num_proposal_iterations: int = 1,
                 use_same_proposal_network: bool = False,
                 proposal_net_args_list: List[Dict] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 num_samples: Optional[int] = None,
                 single_jitter: bool = False,
                 proposal_warmup: int = 5000,
                 proposal_update_every: int = 5,
                 use_proposal_weight_anneal: bool = True,
                 proposal_weights_anneal_max_num_iters: int = 1000,
                 proposal_weights_anneal_slope: float = 10.0,
                 # appearance embedding (phototourism)
                 use_appearance_embedding: bool = False,
                 appearance_embedding_dim: int = 0,
                 num_images: Optional[int] = None,
                 **kwargs,
                 ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.cuda_ray = False
        self.taichi_ray = False
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)

        self.field = KPlaneField(
            aabb,
            grid_config=self.config,
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
        )
        #TODO bg_net
        num_layers_bg = 3  # 3 in paper
        hidden_dim_bg = 32  # 64 in paper
        encoding = 'frequency_torch' # pure pytorch
        self.num_layers_bg = num_layers_bg
        self.hidden_dim_bg = hidden_dim_bg
        self.encoder_bg, self.in_dim_bg = get_encoder(encoding, input_dim=3, multires=4)
        self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)

        # Initialize proposal-sampling nets
        self.density_fns = []
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = torch.nn.ModuleList()
        if use_same_proposal_network:
            assert len(self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)])
        else:
            for i in range(self.num_proposal_iterations):
                prop_net_args = self.proposal_net_args_list[min(i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for network in self.proposal_networks])

        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.proposal_warmup], [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )

    def background(self, d):

        h = self.encoder_bg(d)  # [N, C]

        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs
    def step_before_iter(self, step):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def forward(self, rays_o, rays_d, bg_color,near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions are 3D.
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, timestamps=timestamps, density_fns=self.density_fns)

        field_out = self.field(ray_samples.get_positions(), ray_bundle.directions, timestamps)
        rgb, density = field_out["rgb"], field_out["density"]

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        #TODO:add bg_color
        bg_color = self.background(rays_d)  # [N, 3]

        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)
        depth = self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        accumulation = self.render_accumulation(weights=weights)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    def get_params(self, lr: float):
        model_params = self.field.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_params["field"] + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]
        bg_net_params = [p for p in self.bg_net.parameters()]

        # field_params = list(set(field_params))
        # nn_params = list(set(nn_params))
        # other_params = list(set(other_params))

        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
            {"params":bg_net_params,"lr":lr}
        ]
