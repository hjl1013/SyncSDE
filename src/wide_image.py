import os
import argparse
import numpy as np
import torch
import warnings
from diffusers import DDIMScheduler
from syncsde.pipelines.pipeline_wide_image import WideImagePipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

PATCH_HEIGHT = PATCH_WIDTH = 512

def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--results_folder', type=str, default='output/wide_image')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-diffusion-2-base')
    parser.add_argument('--negative_guidance_scale', default=7.5, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--inv_lambda', default=5.0, type=float)
    parser.add_argument('--n_patches', default=13, type=int)
    parser.add_argument('--init_xt_from_zt', action='store_true')

    args = parser.parse_args()
    seed_everything()
   
    if torch.cuda.is_available():
        device = f"cuda"
    else:
        device = "cpu"
    
    os.makedirs(args.results_folder, exist_ok=True)
    
    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Make the editing pipeline
    pipe = WideImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    _prompt_str = open(args.prompt_path).read().split('\n')[0]

    seed = args.random_seed
    BATCH_SIZE = args.n_patches 
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt_str = [_prompt_str for _ in range(BATCH_SIZE)]
    neg_prompt = ["" for _ in range(BATCH_SIZE)]

    IMG_WIDTH = PATCH_WIDTH + (BATCH_SIZE - 1) * PATCH_WIDTH // 4
    IMG_HEIGHT = PATCH_HEIGHT

    mask_size = PATCH_HEIGHT // 8
    mask_step = mask_size // 4
    mask = torch.zeros(size=(BATCH_SIZE, 4, mask_size, mask_size,), device=device)
    mask[:, :, :, mask_size - mask_step:mask_size] += 1.0

    patch_img_pil, mask_out, result_pil = pipe(
        prompt=prompt_str,
        num_inference_steps=args.num_ddim_steps,
        generator=generator,
        height=PATCH_HEIGHT,
        width=PATCH_WIDTH, 
        mask=mask,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt=neg_prompt, 
        pano_width=IMG_WIDTH,
        pano_height=IMG_HEIGHT,
        inv_lambda=args.inv_lambda,
        n_patches=args.n_patches,
        init_xt_from_zt=args.init_xt_from_zt
    )
    
    result_pil[0].save(os.path.join(args.results_folder, f"output_wide_image.png"))
