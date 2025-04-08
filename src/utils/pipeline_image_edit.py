import sys
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
sys.path.insert(0, "src/utils")
from base_pipeline import BasePipeline
from cross_attention import prep_unet_modified

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class RealImageEditPipeline(BasePipeline):
    def first_stage(
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
        task_name: Optional[str] = None,
        mask_res: Optional[int] = None,
        tgt_word_start_idx: Optional[int] = None,
        mask_thres: Optional[float] = None,
    ):
        x_in.to(dtype=self.unet.dtype, device=self._execution_device)
        self.src_word = task_name.split("2")[0].split()
        self.tgt_word = task_name.split("2")[1].split()
        
        # 0. modify the unet to be useful :D
        self.unet = prep_unet_modified(self.unet)

        # 1. setup all caching objects
        d_ref_t2attn = {} # reference cross attention maps
        d_ref_t2attn_for_mask = {} # reference cross attention maps

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
        x_in = x_in.to(dtype=self.unet.dtype, device=self._execution_device)

        # 3. Encode input prompt = 2x77x1024
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,)
        src_token_pos = [tgt_word_start_idx + 1]
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels

        # randomly sample a latent code if not provided
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, x_in,)

        latents_init = latents.clone()
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. First Denoising loop for getting the reference cross attention maps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        x_src_t_list = []
        mask_target_t_list = []
        self_attn_t_list = []
        with torch.no_grad():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    x_src_t_list.append(latents)
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample

                    # add mask_target_t refer to Prompt-to-Prompt as
                    # https://github.com/google/prompt-to-prompt/blob/main/prompt-to-prompt_stable.ipynb
                    temp_list = []
                    d_ref_t2attn[t.item()] = {}
                    d_ref_t2attn_for_mask[t.item()] = {}
                    for module in self.unet.named_children():
                        if 'down' in module[0] or 'up' in module[0] or 'mid' in module[0]:
                            for name, sub_module in module[1].named_modules():
                                sub_module_name = type(sub_module).__name__
                                if sub_module_name == "CrossAttention" and 'attn2' in name:
                                    # add the cross attention map to the dictionary
                                    attn_mask = sub_module.attn_probs # size is num_channel,s*s,77
                                    d_ref_t2attn[t.item()][f"{module[0]}.{name}"] = attn_mask.detach().cpu()
                                    
                                    if attn_mask.shape[1] <= mask_res ** 2 :
                                        if attn_mask.shape[1] == mask_res ** 2 :
                                            attn_mask = attn_mask.reshape(1, -1, mask_res, mask_res, attn_mask.shape[-1])[0]
                                        else:
                                            length = int(np.sqrt(attn_mask.shape[1]))
                                            attn_mask = attn_mask.reshape(-1, length, length, attn_mask.shape[-1])
                                            attn_mask = attn_mask.permute(0, 3, 1, 2)
                                            attn_mask = torch.nn.functional.interpolate(attn_mask, (mask_res, mask_res), mode='bicubic', align_corners=False)
                                            attn_mask = attn_mask.permute(0, 2, 3, 1)
                                        _attn_mask = torch.mean(torch.stack([attn_mask[:, :, :, src_token_pos[i]] for i in range(len(src_token_pos))]), dim = 0)
                                        temp_list.append(_attn_mask)
                                
                    mask_target_t = torch.cat(temp_list, 0)
                    mask_target_t = mask_target_t.sum(0) / mask_target_t.shape[0]
                    mask_target_t = mask_target_t / mask_target_t.max()
                    mask_target_t = mask_target_t.unsqueeze(-1).expand(*mask_target_t.shape, latents.shape[1])
                    mask_target_t = mask_target_t.permute(2,0,1)
                    mask_target_t = mask_target_t.unsqueeze(0)
                    mask_target_t = torch.nn.functional.interpolate(mask_target_t, (latents.shape[2], latents.shape[3]), mode='bicubic', align_corners=False)
                    
                    #mask_target_t = torch.clamp(mask_target_t, min=0.0, max=1.0)
                    mask_target_t_list.append(mask_target_t.detach().cpu())
                    
                    # self attention-map
                    self_attn_mask = self.unet.up_blocks[-1].attentions[-1].transformer_blocks[0].attn1.attn_probs
                    _self_attn_mask = torch.mean(self_attn_mask, 0)
                    self_attn_t_list.append(_self_attn_mask)
                    

                    # classifier-free guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        # make the reference image (reconstruction)
        with torch.no_grad():
            image_rec = self.decode_latents(latents.detach())

        # 9. Run safety checker (skip)
        image_rec, has_nsfw_concept = self.run_safety_checker(image_rec, device, prompt_embeds.dtype)


        # 10. Convert to PIL
        image_rec = self.numpy_to_pil(image_rec)

        # 11. Mask generation        
        mask_target = sum(mask_target_t_list) / len(mask_target_t_list)
        mask_target = mask_target.to(device)
        mask_target = torch.clamp(mask_target, min=0.0, max=1.0)
        mask_target = mask_target.reshape([mask_target.shape[0], mask_target.shape[1], -1]) 
        
        self_attn_mask = sum(self_attn_t_list) / len(self_attn_t_list)
        self_attn_mask = self_attn_mask.to(device)

        mask_target_new = torch.einsum('ij,bcj->bci', self_attn_mask, mask_target) 
        mask_target_new = mask_target_new.reshape([mask_target_new.shape[0], mask_target_new.shape[1],
                                                   latents.shape[2], latents.shape[3]])
        mask_target = mask_target.reshape([mask_target.shape[0], mask_target.shape[1], latents.shape[2], latents.shape[3]])
        mask_target_new = torch.where(mask_target_new > mask_thres, 1., 0.) 
        
        mask_torch = mask_target_new.detach().clone()
        mask_target_new = mask_target_new[:, 0:1, :, :]
        mask_target_new = mask_target_new.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        mask_target_new = self.numpy_to_pil(mask_target_new)

        return image_rec, mask_target_new, mask_torch
    
    
    def second_stage(
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
        inv_lambda: Optional[float] = 0.1,
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

        latents_init = latents.clone()
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
                    
                    latents[1:2] = latents[1:2] - gamma_t * (1.0 - mask) * inv_lambda * (t.item() / T) * (old_latents[1:2] - old_latents[0:1])
                    latents[2:] = latents[2:] - gamma_t * ((1.0 - mask) * inv_lambda * (t.item() / T) * (old_latents[2:] - old_latents[0:1]) + \
                                                            mask * inv_lambda * (t.item() / T) * (old_latents[2:] - old_latents[1:2]))

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
    