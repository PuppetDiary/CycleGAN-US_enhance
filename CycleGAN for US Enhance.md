# CycleGAN for US Enhance

This code is a secondary development based on the official [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.

We introduce an additional model branch integrated into the original CycleGAN pipeline. This branch adopts either a ResNet18 or ViT-based architecture and serves to evaluate the effectiveness of different pretrained weights—trained on ultrasound datasets—when applied to ultrasound image generation. The branch follows an encoder-decoder structure, and two branch options are provided: `resnet18_branch` and `vit_base_branch`. Pretrained weights can be loaded into the encoder of each branch. For instance, in our work, models such as USFM, DINOv2, and MAE—trained using a ViT backbone on ultrasound datasets—are loaded into the `vit_base_branch` for comparative analysis.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https:xxx
cd xxx
```

### Dataset

You can download datasets [here](https://ultrasoundenhance2023.grand-challenge.org/datasets/).

### Train

- Train model with USFM weights:

```python
python train.py --dataroot ./datasets/train_datasets/breast --name breast_cyclegan_USFM --model cycle_gan --netG_branch vit_base_branch --pretrained_dir ./weights/USFM.ckpt --pretrained_USFM
```

- Train model with custom resnet18 weights:

```python
python train.py --dataroot ./datasets/train_datasets/breast --name breast_cyclegan_resnet18 --model cycle_gan --netG_branch vit_base_branch --pretrained_dir ./weights/Resnet18.ckpt --pretrained_OwnResnet18
```

- Check options file for more supported options.

### Test

Please specific **--name** option where you have saved pretrained weights in training stage.

- Test model which trained with USFM weights:

```python
python test.py --dataroot ./datasets/train_datasets/breast --name breast_cyclegan_USFM --model cycle_gan --netG_branch vit_base_branch
```

- Test model which trained with custom resnet18 weights:

```python
python test.py --dataroot ./datasets/train_datasets/breast --name breast_cyclegan_resnet18 --model cycle_gan --netG_branch vit_base_branch
```

## Citation

If you use this code for your research, please cite our papers.

## Acknowledgments

Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).