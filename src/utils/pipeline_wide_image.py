import sys
import torch
from typing import Any, Dict, List, Optional, Union
sys.path.insert(0, "src/utils")
from base_pipeline import BasePipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class WideImagePipeline(BasePipeline):
    def patch_mapping(self, latents, j):
        h = latents.shape[-1]
        _y = torch.zeros_like(latents[0:1])
        _y[:, :, :, 0:3*h//4] = latents[j-1:j, :, :, h//4:h].detach().clone()
        return _y

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        x_in: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
        pano_width: Optional[int] = None,
        pano_height: Optional[int] = None,
        n_patches: Optional[int] = None,
        init_xt_from_zt: Optional[bool] = None,
        inv_lambda: Optional[float] = 0.1,
    ):
        # 1. setup all caching objects

        # 2. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt = 2x77x1024
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,)
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels

        # randomly sample a latent code if not provided
        if not init_xt_from_zt:
            latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, x_in,)
        else:
            # print("init xt from zt")
            latents_zt = self.prepare_latents(1, num_channels_latents, pano_height, pano_width, prompt_embeds.dtype, device, generator, x_in,)
            _latents = []
            for j in range(n_patches):
                _latents.append(latents_zt[:, :, :, 16 * j: 16 * j + 64])
            latents = torch.cat(_latents, dim=0)

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop 
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        T = max(timesteps).item()
        with torch.no_grad():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    old_latents = latents.clone()
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample
                    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    prev_t = t - self.scheduler.config.num_train_timesteps // num_inference_steps
                    alpha_cur = self.scheduler.alphas_cumprod[t]
                    alpha_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
                    gamma_t = (alpha_prev / alpha_cur).sqrt() - ((1.0 - alpha_prev) / (1.0 - alpha_cur)).sqrt()

                    for j in range(1, batch_size):
                        latents[j:j+1] = latents[j:j+1] - gamma_t * (1.0 - mask[j:j+1]) * inv_lambda * (t.item() / T) * (old_latents[j:j+1] - self.patch_mapping(old_latents, j))

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()


        # 8. Post-processing
        with torch.no_grad():
            latent_height = pano_height // 8
            latent_width = pano_width // 8
            latent_step = latent_height // 4 

            final_latents = torch.zeros(size=(1, 4, latent_height, latent_width)).to(device=latents.device, dtype=latents.dtype)
            final_latents[:, :, :, :latent_height] = latents[0:1, :, :, :latent_height]
            for i in range(batch_size - 1):
                final_latents[:, :, :, latent_step * (i+1) : latent_height + latent_step * (i+1)] = latents[i+1 : i+2, :, :, :latent_height]
            image = self.decode_latents(latents.detach())
            wide_image = self.decode_latents(final_latents.detach())
            mask = mask[:, 0:1, :, :]
            mask = mask.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            

        # 9. Run safety checker: skip in the implementation step
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        patch_images = self.numpy_to_pil(image)
        result = self.numpy_to_pil(wide_image)
        mask = self.numpy_to_pil(mask)
        
        return patch_images, mask, result
    
