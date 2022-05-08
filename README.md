# Shape-Completion-using-IMLE

This repository provides PyTorch implementation of our paper:

[Multimodal Shape Completion via IMLE](
https://arxiv.org/abs/2106.16237)


## Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.6



## Dependencies

1. Install python package dependencies through pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Install external dependency [PyTorchEMD](https://github.com/daerduoCarey/PyTorchEMD) for the usage of EMD loss.



## Data

We use three datasets in our paper.

1. 3D-EPN

   Please download the partial scan point cloud data from [their website](http://kaldir.vc.in.tum.de/adai/CNNComplete/shapenet_dim32_sdf_pc.zip) and extract it into `data` folder. For the complete point clouds data, please download it from [PKU disk](https://disk.pku.edu.cn:443/link/9A3E1AC9FBA4DEBD705F028650CBE8C7) and extract it into `data` folder. Or you can follow this [blender-render script](https://github.com/xuelin-chen/blender_renderer) to virtually scan ShapeNet objects by yourself.



## Training

Training scripts can be found in `scripts` folder and please see `common.py` for specific definition of all command line parameters. For example, to train on 3DEPN chair category:

```bash
# 1. pre-train the PointNet AE
sh scripts/3depn/chair/train-3depn-chair-ae.sh

# 2. train the conditional IMLE for multimodal shape completion
sh scripts/3depn/chair/train-3depn-chair-imle.sh

```

Training log and model weights will be saved in `proj_log` folder by default. 



## Testing

Testing scripts can also be found in `scripts` folder. For example, to test the model trained on 3DEPN chair category:

```bash
# by default, run over all test examples and output 10 completion results for each
sh scripts/3depn/chair/test-3depn-chair-gan.sh
```

The completion results, along with the input parital shape, will be saved in `proj_log/mpc-3depn-chair/gan/results` by default. 



## Evaluation

Evaluation code can be found in `evaluation` folder. To evaluate the completion *diversity* , *fidelity* and *quality*:

```bash
cd evaluation/
# calculate Total Mutual Difference (TMD)
python total_mutual_diff.py --src {path-to-saved-testing-results}
# calculate Unidirectional Hausdorff Distance (UHD) and completeness
python completeness.py --src {path-to-saved-testing-results}
# calculate Minimal Matching Distance (MMD), this requries a tensorflow environment
python mmd.py --src {path-to-saved-testing-results} --dataset {which-dataset} --class_name {which-category} -g 0
```

Note that MMD calculation requires an compiled `external` library from [its original repo](https://github.com/optas/latent_3d_points).



## Pre-trained models

The code is borrowed from https://github.com/ChrisWu1997/Multimodal-Shape-Completion 



## Cite

Please cite our work if you find it useful:

```
@article{arora2021multimodal,
  title={Multimodal Shape Completion via IMLE},
  author={Arora, Himanshu and Mishra, Saurabh and Peng, Shichong and Li, Ke and Mahdavi-Amiri, Ali},
  journal={arXiv preprint arXiv:2106.16237},
  year={2021}
}
```

