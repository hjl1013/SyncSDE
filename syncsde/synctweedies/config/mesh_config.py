import argparse 

def load_mesh_config():
    parser = argparse.ArgumentParser()
    
    # File Config
    parser.add_argument('--mesh_config_relative', action='store_true', help="Search mesh file relative to the config path instead of current working directory")
    parser.add_argument('--save_dir_now', action='store_true')

    # Diffusion Config
    parser.add_argument('--negative_prompt', type=str, default='oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=15.5, help='Recommend above 12 to avoid blurriness')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)

    # ControlNet Config
    parser.add_argument('--model', type=str, default='controlnet')
    parser.add_argument('--cond_type', type=str, default='depth', help='Support depth and normal, less multi-face in normal mode, but some times less details')
    parser.add_argument('--guess_mode', action='store_true')
    parser.add_argument('--conditioning_scale', type=float, default=0.7)
    parser.add_argument('--conditioning_scale_end', type=float, default=0.9, help='Gradually increasing conditioning scale for better geometry alignment near the end')
    parser.add_argument('--control_guidance_start', type=float, default=0.0)
    parser.add_argument('--control_guidance_end', type=float, default=0.99)
    parser.add_argument('--guidance_rescale', type=float, default=0.0, help='Not tested')

    # Multi-View Config
    parser.add_argument('--latent_view_size', type=int, default=96, help='Larger resolution, less aliasing in latent images; quality may degrade if much larger trained resolution of networks')
    parser.add_argument('--latent_tex_size', type=int, default=1536, help='Originally 1536 in paper, use lower resolution save VRAM')
    parser.add_argument('--rgb_view_size', type=int, default=768)
    parser.add_argument('--rgb_tex_size', type=int, default=1024)
    parser.add_argument('--camera_azims', type=int, action='append', default=[-180, -135, -90, -45, 0, 45, 90, 135])
    parser.add_argument('--top_cameras', action='store_false', help='Two cameras added to paint the top surface')
    parser.add_argument('--mvd_end', type=float, default=0.8, help='Time step to stop texture space aggregation')
    parser.add_argument('--ref_attention_end', type=float, default=0.2, help='Lower->better quality; higher->better harmonization')
    parser.add_argument('--shuffle_bg_change', type=float, default=0.4, help='Use only black and white background after certain timestep')
    parser.add_argument('--shuffle_bg_end', type=float, default=0.8, help='Don\'t shuffle background after certain timestep. background color may bleed onto object')
    parser.add_argument('--mesh_scale', type=float, default=1.0, help='Set above 1 to enlarge object in camera views')
    parser.add_argument('--mesh_autouv', action='store_true', help='Use Xatlas to unwrap UV automatically')
    parser.add_argument("--noise_space", type=str, default="texture")

    # Logging Config
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--view_fast_preview', action='store_false')
    parser.add_argument('--tex_fast_preview', action='store_false')

    # SDEdit Config
    parser.add_argument('--sdedit', action='store_true')
    parser.add_argument('--sdedit_prompt', type=str, default=None)
    parser.add_argument('--sdedit_timestep', type=float, default=0.2) # 0.0 = x_T (Full generation) / 1.0 = x_0 (No generation)
    parser.add_argument('--log_height', type=int, default=256)
    parser.add_argument('--log_width', type=int, default=256)
    parser.add_argument('--max_batch_size', type=int, default=48)
    parser.add_argument('--disable_voronoi', action='store_true')
    parser.add_argument('--sampling_method', type=str, default="ddpm")
    parser.add_argument('--rasterize_mode', type=str, default="nearest")
    parser.add_argument('--initialize_xt_from_zt', action='store_true')
    parser.add_argument('--save_gif', action='store_true')

    # syncsde config
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--results_folder', type=str, default='output/mesh_texturing')
    parser.add_argument('--inv_lambda', default=5.0, type=float)
    
    
    options = parser.parse_args()
    
    return options