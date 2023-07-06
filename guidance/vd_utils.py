from diffusers.utils import export_to_video
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, \
    TextToVideoSDPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)

        # print("forward-gt_grad\n",gt_grad)

        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        #print(gt_grad)
        gt_grad = gt_grad * grad_scale
        # print("backward-grad_scale:\n",grad_scale)
        # print("backward-gt_grad:\n",gt_grad)
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class VideoDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, model_key="damo-vilab/text-to-video-ms-1.7b", t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device

        print(f'[INFO] loading stable diffusion...')


        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        self.pipe = TextToVideoSDPipeline.from_pretrained(
            model_key, torch_dtype=self.precision_t , variant="fp16"
        )

        if vram_O:
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            #pipe.enable_xformers_memory_efficient_attention()
            #pipe.enable_var_tiling()
            # pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet

        self.scheduler = self.pipe.scheduler
        #
        # del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(50)
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def train_step(self, ratio, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path: Path = None):
        pred_rgb = pred_rgb.squeeze().permute(0, 3, 1, 2).contiguous()
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)
        latents = latents.permute(1,0,2,3).contiguous()
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1,[1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.rand(latents.shape[2:])
            noise = noise.repeat(latents.shape[0], latents.shape[1], 1, 1).to(self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.stack([latents_noisy] * 2,dim=0)
            tt = torch.stack([t] * 2,dim=0)
            #sample(`torch.FloatTensor`): (batch, num_frames, channel, height, width) noisy inputs tensor
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1
                b = len(noise_pred)
                a_t = alphas[index].reshape(b, 1, 1, 1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b, 1, 1, 1)).to(self.device)
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()  # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # TODO: also denoise all-the-way

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image], dim=0)
                save_image(viz_images, save_guidance_path)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    def train_step_perpneg(self, ratio, text_embeddings, weights, pred_rgb, guidance_scale=100, as_latent=False,
                           grad_scale=1,
                           save_guidance_path: Path = None):
        # pred_rgb [num,h,w,3]
        B = pred_rgb.shape[1]
        pred_rgb = pred_rgb.squeeze().permute(0, 3, 1, 2).contiguous()
        K = (text_embeddings.shape[0] // B) - 1  # maximum number of prompts
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)
        latents = latents.permute(1,0,2,3).contiguous()

        current_max = min(int((self.max_step - self.min_step) * ratio) + self.min_step + 1, self.max_step)
        if ratio > 0.5:
            t = torch.randint(self.min_step, current_max, [1],dtype=torch.long, device=self.device)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        else:
            t = torch.randint(self.min_step, self.max_step + 1,[1],dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.rand(latents.shape[2:])
            noise = noise.repeat(latents.shape[0],latents.shape[1], 1, 1).to(self.device)


            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.stack([latents_noisy] * (1+K),dim=0)
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds,
                                                                                                weights, B)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # [1,4,64,64]
        # self.writer.add_image("grad", grad, global_step=self.global_step, dataformats='NCHW')
        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1
                b = len(noise_pred)
                a_t = alphas[index].reshape(b, 1, 1, 1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b, 1, 1, 1)).to(self.device)
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()  # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image], dim=0)
                save_image(viz_images, save_guidance_path)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)
        # print("we did it")
        return loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents



if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--negative', default='', type=str)

    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_frames', type=int, default=16)

    parser.add_argument('--video', action='store_true', help="use video diffusion ")
    parser.add_argument('--vh', type=int, default=512, help="render height for NeRF in video training")
    parser.add_argument('--vw', type=int, default=512, help="render height for NeRF in video training")
    parser.add_argument('--model_key', type=str, default='damo-vilab/text-to-video-ms-1.7b', help="hugging face Video diffusion model key")
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    vd = VideoDiffusion(device, opt.fp16, opt.vram_O, opt.model_key)

    imgs = vd.pipe(prompt=opt.prompt,negative_prompt=opt.negative,height=opt.vh,width=opt.vw,num_frames=opt.num_frames).frames
    video_path = export_to_video(imgs,output_video_path='./test.mp4')




