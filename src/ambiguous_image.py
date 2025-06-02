import os
import argparse
import numpy as np
import torch
import warnings
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from diffusers import DDIMScheduler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def skew(latent, skew_factor=-1.5):
    assert len(latent.shape) == 3  # (C, H, W)
    c, h, w = latent.shape
    h_center = h // 2

    cols = []
    for j in range(w):
        d = int(skew_factor * (j - h_center))
        col = latent[:, :, j]
        cols.append(col.roll(d, dims=1))
    
    skewed_latent = torch.stack(cols, dim=2)
    return skewed_latent

def rotate(image, rot_factor):
    # Convert to numpy array, rotate, and convert back
    image_np = image.permute(1, 2, 0).numpy()  # Convert to HWC
    rotated_image = np.rot90(image_np, k=rot_factor).copy()  # Rotate
    rotated_image = torch.tensor(rotated_image).permute(2, 0, 1)  # Convert back to CHW
    return rotated_image

def flip(image):
    image_np = image.permute(1, 2, 0).numpy()  # Convert to HWC
    flipped_image = np.flip(image_np, 0).copy()
    flipped_image = torch.tensor(flipped_image).permute(2, 0, 1)
    return flipped_image


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, required=True) # default='/home/hyunjun/projects/CVPR2025/quant_new/prompts/ambiguous_images_prompts.yml', type=str)
    parser.add_argument('--transform', type=str, required=True, choices=['rotate_cw', 'rotate_ccw', 'rotate_180', 'skew', 'flip'])
    parser.add_argument('--results_folder', type=str, default='output/ambiguous_image')
    parser.add_argument('--num_ddim_steps', type=int, default=30)
    parser.add_argument('--negative_guidance_scale', default=10.0, type=float)  
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--inv_lambda', default=5.0, type=float)
    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)
    generator = torch.manual_seed(args.random_seed)

    prompt_str = open(args.prompt_path).read().split('\n')
    prompt1 = prompt_str[0]; prompt2 = prompt_str[1]

    with torch.no_grad():   
        if args.transform == "rotate_cw" or args.transform == "rotate_ccw" or args.transform == "rotate_180":
            from syncsde.pipelines.pipeline_ambiguous_rotation_df import IFPieplineEditStage_1 as RotateIFPieplineEditStage_1
            from syncsde.pipelines.pipeline_ambiguous_rotation_df import IFPieplineEditStage_2 as RotateIFPieplineEditStage_2
            rotate_stage_1 = RotateIFPieplineEditStage_1.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", 
                variant="fp16", 
                torch_dtype=torch.float16,
            ).to(device)
            # rotate_stage_1.enable_model_cpu_offload()

            # stage 2
            rotate_stage_2 = RotateIFPieplineEditStage_2.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            ).to(device)
            # rotate_stage_2.enable_model_cpu_offload()

            scheduler = DDIMScheduler.from_config(rotate_stage_1.scheduler.config)
            rotate_stage_1.scheduler = rotate_stage_2.scheduler = scheduler
            rotate_stage_1.scheduler.set_timesteps(
                args.num_ddim_steps, device=device
            )
            
            prompt_embeds = [rotate_stage_1.encode_prompt(prompt1), rotate_stage_1.encode_prompt(prompt2)]
            prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
            prompt_embeds = torch.cat(prompt_embeds)
            negative_prompt_embeds = torch.cat(negative_prompt_embeds)
            
            do_ccw = False if args.transform == "rotate_cw" else True
            double_rotate = True if args.transform == "rotate_180" else False

            img_pt_stage_1 = rotate_stage_1(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=args.num_ddim_steps,
                guidance_scale=args.negative_guidance_scale,
                generator=generator,
                output_type="pt",
                clean_caption=False,
                inv_lambda=args.inv_lambda,
                do_ccw=do_ccw,
                double_rotate=double_rotate
            )

            img_pil_stage_2 = rotate_stage_2(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=img_pt_stage_1.detach().clone(),
                num_inference_steps=args.num_ddim_steps,
                guidance_scale=args.negative_guidance_scale,
                generator=generator,
                output_type="pil",
                clean_caption=False
            )

            prompt2_image = pil_to_tensor(img_pil_stage_2[1]).float() / 255.0
            if args.transform == "rotate_cw":
                prompt1_image = rotate(prompt2_image, 1)
            elif args.transform == "rotate_ccw":
                prompt1_image = rotate(prompt2_image, -1)
            elif args.transform == "rotate_180":
                prompt1_image = rotate(prompt2_image, 2)

        elif args.transform == "skew":
            from syncsde.pipelines.pipeline_ambiguous_skew_df import IFPieplineEditStage_1 as SkewIFPieplineEditStage_1
            from syncsde.pipelines.pipeline_ambiguous_skew_df import IFPieplineEditStage_2 as SkewIFPieplineEditStage_2
            skew_stage_1 = SkewIFPieplineEditStage_1.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", 
                variant="fp16", 
                torch_dtype=torch.float16,
            ).to(device)
            # skew_stage_1.enable_model_cpu_offload()

            # stage 2
            skew_stage_2 = SkewIFPieplineEditStage_2.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            ).to(device)
            # skew_stage_2.enable_model_cpu_offload()

            scheduler = DDIMScheduler.from_config(skew_stage_1.scheduler.config)
            skew_stage_1.scheduler = skew_stage_2.scheduler = scheduler
            skew_stage_1.scheduler.set_timesteps(
                args.num_ddim_steps, device=device
            )

            prompt_embeds = [skew_stage_1.encode_prompt(prompt1), skew_stage_1.encode_prompt(prompt2)]
            prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
            prompt_embeds = torch.cat(prompt_embeds)
            negative_prompt_embeds = torch.cat(negative_prompt_embeds)
            
            img_pt_stage_1 = skew_stage_1(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=args.num_ddim_steps,
                guidance_scale=args.negative_guidance_scale,
                generator=generator,
                output_type="pt",
                clean_caption=False,
                inv_lambda=args.inv_lambda
            )

            img_pil_stage_2 = skew_stage_2(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=img_pt_stage_1.detach().clone(),
                num_inference_steps=args.num_ddim_steps,
                guidance_scale=args.negative_guidance_scale,
                generator=generator,
                output_type="pil",
                clean_caption=False
            )

            prompt2_image = pil_to_tensor(img_pil_stage_2[1]).float() / 255.0
            prompt1_image = skew(prompt2_image)


        elif args.transform == "flip":
            from syncsde.pipelines.pipeline_ambiguous_ver_flip_df import IFPieplineEditStage_1 as FlipIFPieplineEditStage_1
            from syncsde.pipelines.pipeline_ambiguous_ver_flip_df import IFPieplineEditStage_2 as FlipIFPieplineEditStage_2
            flip_stage_1 = FlipIFPieplineEditStage_1.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", 
                variant="fp16", 
                torch_dtype=torch.float16,
            ).to(device)
            # flip_stage_1.enable_model_cpu_offload()

            # stage 2
            flip_stage_2 = FlipIFPieplineEditStage_2.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            ).to(device)
            # flip_stage_2.enable_model_cpu_offload()

            scheduler = DDIMScheduler.from_config(flip_stage_1.scheduler.config)
            flip_stage_1.scheduler = flip_stage_2.scheduler = scheduler
            flip_stage_1.scheduler.set_timesteps(
                args.num_ddim_steps, device=device
            )

            prompt_embeds = [flip_stage_1.encode_prompt(prompt1), flip_stage_1.encode_prompt(prompt2)]
            prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
            prompt_embeds = torch.cat(prompt_embeds)
            negative_prompt_embeds = torch.cat(negative_prompt_embeds)

            img_pt_stage_1 = flip_stage_1(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=args.num_ddim_steps,
                guidance_scale=args.negative_guidance_scale,
                generator=generator,
                output_type="pt",
                clean_caption=False,
                inv_lambda=args.inv_lambda
            )

            img_pil_stage_2 = flip_stage_2(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=img_pt_stage_1.detach().clone(),
                num_inference_steps=args.num_ddim_steps,
                guidance_scale=args.negative_guidance_scale,
                generator=generator,
                output_type="pil",
                clean_caption=False
            )

            prompt2_image = pil_to_tensor(img_pil_stage_2[1]).float() / 255.0
            prompt1_image = flip(prompt2_image)
        
        else:
            raise Exception("Not a considered transformation")


        prompt1_image = (prompt1_image * 255).clamp(0, 255).byte()  # Denormalize to [0, 255]
        prompt1_image = to_pil_image(prompt1_image)
        prompt2_image = (prompt2_image * 255).clamp(0, 255).byte()  # Denormalize to [0, 255]
        prompt2_image = to_pil_image(prompt2_image)

        prompt1_image.save(os.path.join(args.results_folder, "prompt_1.png"))
        prompt2_image.save(os.path.join(args.results_folder, "prompt_2.png"))

    

