# SyncSDE: A Probabilistic Framework for Diffusion Synchronization

This repository includes official implementation of "SyncSDE: A Probabilistic Framework for Diffusion Synchronization" (CVPR 2025).

## Installation

This repository is tested with Python 3.9, CUDA 11.8.

    pip install -r requirements.txt

If error occurs on numpy version, please re-install numpy via

    conda install -c conda-forge numpy

and upgrade to version 1.22.4.

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
                        --results_folder "output/real_image_editing" \
                        --use_float_16
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

**[NOTE]** You may upgrade `transformer` version to use `fp16` variant of the pretrained DeepFloyd-IF model.

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


This repository is constructed based on [Conditional Score Guidance](https://github.com/Hleephilip/CSG) and [DeepFloyd-IF](https://github.com/deep-floyd/IF). The source image for text-driven real image editing is brought from the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset.
