import sys
import torch
from typing import Any, Dict, List, Optional, Union

from syncsde.pipelines.base_pipeline import BasePipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class MaskT2IPipeline(BasePipeline):
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
        inv_lambda: Optional[float] = None,
    ):
        x_in.to(dtype=self.unet.dtype, device=self._execution_device)

        # 1. Default height and width to unet
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
        x_in = x_in.to(dtype=self.unet.dtype, device=self._execution_device)

        # 3. Encode input prompt = 2x77x1024
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,)
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels

        # randomly sample a latent code if not provided
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, x_in,)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
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

                    latents[1:2] = latents[1:2] - gamma_t * (1.0 - mask[1:2]) * inv_lambda * (t.item() / T) * (old_latents[1:2] - old_latents[0:1])
                    latents[2:] = latents[2:] - gamma_t * ((1.0 - mask[1:2]) * inv_lambda * (t.item() / T) * (old_latents[2:] - old_latents[0:1]) + \
                                                            mask[1:2] * inv_lambda * (t.item() / T) * (old_latents[2:] - old_latents[1:2]))

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()


        # 8. Post-processing
        with torch.no_grad():
            image = self.decode_latents(latents.detach())

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        image = self.numpy_to_pil(image)
        
        return image
    

