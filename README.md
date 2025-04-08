# SyncSDE: A Probabilistic Framework for Diffusion Synchronization

This repository includes official implementation of "SyncSDE: A Probabilistic Framework for Diffusion Synchronization" (CVPR 2025).

## Installation

TBD


## Running SyncSDE


### Mask-based Text-to-Image Generation

```
python src/mask_based_T2I.py --prompt_path "data/prompt_mask_T2I.txt" \
                             --mask_path "data/mask_sample.png" \
                             --results_folder "./output/mask_T2I" \
                             --random_seed 0 --use_float_16 --inv_lambda 5.0
```


### Text-driven Real Image Editing

First invert the real image using DDIM inversion.

```
python src/inversion.py --input_image "data/image.png" \
                        --results_folder "output/real_image_editing"
```

Then edit the real image.


```
python src/real_image_editing.py --inversion "./output/real_image_editing/inversion/image.pt" \
                                 --prompt "./output/real_image_editing/prompt/image.txt" \
                                 --results_folder "./output/real_image_editing" --task_name "cat2dog" \
                                 --random_seed 0 --use_float_16 --inv_lambda 5.0 
```

### Wide Image Generation

```
python src/wide_image.py --prompt_path "./data/prompt_wide_image.txt" \
                         --results_folder "output/wide_image" \
                         --n_patches 13 --init_xt_from_zt \
                         --random_seed 0 --use_float_16 --inv_lambda 5.0
```

### Ambiguous Image Generation

To use the pretrained [Deepfloyd-IF](https://github.com/deep-floyd/IF) model, please log in to hugging face by following the [instructions](https://huggingface.co/docs/diffusers/en/api/pipelines/deepfloyd_if).

Then, run the script:

```
python src/ambiguous_image.py --prompt_path "data/prompt_ambiguous.txt" \
                              --transform "rotate_cw" \
                              --results_folder "./output/ambiguous_image" \
                              --random_seed 0 --use_float_16 --inv_lambda 5.0
```

You may use 5 types of transforms: `'rotate_cw'`, `'rotate_ccw'`, `'rotate_180'`, `'skew'`, and `'flip'`.

## Acknowledgments


This repository is constructed based on [Conditional Score Guidance](https://github.com/Hleephilip/CSG) and [DeepFloyd-IF](https://github.com/deep-floyd/IF). The source image for text-driven real image editing is brought from [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero/tree/main) repository. 
