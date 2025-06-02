import os
import argparse
import torch
import glob
import clip
import warnings

from diffusers import DDIMScheduler
from syncsde.pipelines.pipeline_image_edit import RealImageEditPipeline
from syncsde.utils.target_prompt import make_target_prompt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inversion', required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--results_folder', type=str, default='output/real_image_editing')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--mask_res', type=int, default=16)
    parser.add_argument('--task_name', type=str, default='cat2leopard')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--inv_lambda', type=float, default=5.0)
    parser.add_argument('--mask_thres', type=float, default=0.3)

    args = parser.parse_args()
    os.makedirs(os.path.join(args.results_folder, "reconstruction"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "edit"), exist_ok=True)
    os.makedirs(os.path.join(args.results_folder, "mask"), exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)
        
    
    assert (os.path.isdir(args.inversion)==os.path.isdir(args.prompt)), "If the inversion is a folder, the prompt should also be a folder"
    if os.path.isdir(args.inversion):
        l_inv_paths = sorted(glob.glob(os.path.join(args.inversion, "*.pt")))
        l_bnames = [os.path.basename(x) for x in l_inv_paths]
        l_prompt_paths = [os.path.join(args.prompt, x.replace(".pt",".txt")) for x in l_bnames]
    else:
        l_inv_paths = [args.inversion]
        l_prompt_paths = [args.prompt]

    # Make the editing pipeline
    pipe = RealImageEditPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    (src_word, tgt_word) = args.task_name.split("2")
    model, preprocess = clip.load("ViT-B/32", device=device)
    tokenized_src_word = clip.tokenize([src_word]).to(device)
    src_word_features = model.encode_text(tokenized_src_word)

    for inv_path, prompt_path in zip(l_inv_paths, l_prompt_paths):
        prompt_str = open(prompt_path).read().strip()
        tgt_prompt, idx = make_target_prompt(prompt_str, src_word, tgt_word, src_word_features, model, device)

        x_T = torch.load(inv_path).unsqueeze(0)
        x = torch.randn((1, 4, 64, 64), device=device)

        rec_pil, mask_target, mask_torch = pipe.first_stage(
            prompt=prompt_str,
            num_inference_steps=args.num_ddim_steps,
            x_in=x_T,
            guidance_scale=args.negative_guidance_scale,
            negative_prompt=prompt_str, 
            task_name=args.task_name,
            mask_res=args.mask_res,
            tgt_word_start_idx=idx,
            mask_thres=args.mask_thres,
        )

        bname = os.path.basename(inv_path).split(".")[0]
        rec_pil[0].save(os.path.join(args.results_folder, f"reconstruction/{bname}.png"))
        mask_target[0].save(os.path.join(args.results_folder, f"mask/{bname}.png"))

        img_pil = pipe.second_stage(
            prompt=[prompt_str, tgt_prompt, tgt_prompt],
            num_inference_steps=args.num_ddim_steps,
            x_in=torch.cat([x_T, x, x_T]), 
            guidance_scale=args.negative_guidance_scale,
            negative_prompt=[prompt_str, "", prompt_str],
            mask=mask_torch,
            inv_lambda=args.inv_lambda
        )
        img_pil[2].save(os.path.join(args.results_folder, f"edit/{bname}.png"))
    
    