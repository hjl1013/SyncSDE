# SyncSDE: A Probabilistic Framework for Diffusion Synchronization

This repository includes official implementation of "SyncSDE: A Probabilistic Framework for Diffusion Synchronization" (CVPR 2025).

## Installation

```
pip install -r requirements.txt
```


## Running SyncSDE


### Mask-based Text-to-Image Generation



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
                                 --results_folder "./output/real_image_editing" \
                                 --task_name "cat2dog" --inv_lambda 5.0 --use_float_16
```

### Wide Image Generation

```
python src/wide_image.py --prompt_path "./data/prompt_wide_image.txt" --random_seed 0 \
                         --results_folder "output/wide_image" --inv_lambda 5.0 \
                         --n_patches 13 --init_xt_from_zt --use_float_16
```

### Ambiguous Image Generation



## Acknowledgments


This repository is constructed based on [Conditional Score Guidance](https://github.com/Hleephilip/CSG) and [DeepFloyd-IF](https://github.com/deep-floyd/IF). The source image for text-driven real image editing is brought from [pix2pix-zero](https://github.com/pix2pixzero/pix2pix-zero/tree/main). 
