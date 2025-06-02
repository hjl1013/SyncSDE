import os
import argparse
import numpy as np
import torch
import warnings
from PIL import Image
from diffusers import DDIMScheduler
from syncsde.pipelines.pipeline_mask_T2I import MaskT2IPipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def preprocess_mask(mask_path, h, w, device):
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, required=True) 
    parser.add_argument('--mask_path', type=str, required=True) 
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--results_folder', type=str, default='output/mask_T2I')
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-diffusion-2-base')
    parser.add_argument('--negative_guidance_scale', default=7.5, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--inv_lambda', default=5.0, type=float)

    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # make the input noise map
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)

    prompt_str = open(args.prompt_path).read().split('\n')
    mask = preprocess_mask(args.mask_path, 64, 64, device)
    mask = torch.cat([mask, mask, mask])

    BATCH_SIZE = len(prompt_str)
    x = torch.randn((BATCH_SIZE, 4, 64, 64), device=device)
    neg_prompt = ["" for _ in range(BATCH_SIZE)]
        
    # Make the editing pipeline
    pipe = MaskT2IPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    img_pil = pipe(
        prompt=prompt_str,
        num_inference_steps=args.num_ddim_steps,
        x_in=x,
        mask=mask,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt=neg_prompt, # use the empty string for the negative prompt
        inv_lambda=args.inv_lambda
    )

    img_pil[2].save(os.path.join(args.results_folder, fr"result.png"))