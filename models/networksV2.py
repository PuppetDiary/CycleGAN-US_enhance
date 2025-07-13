"""
Neural Network Models Definition Module (networksV2.py)

This module contains generator network implementations based on ResNet and Vision Transformer,
supporting various pretrained weight loading options, primarily used for image-to-image translation tasks.

Main Components:
- ResNet18Encoder: ResNet-18 encoder
- ViTFeatureExtractor: Vision Transformer feature extractor
- BranchResNet: Branch network combining ResNet encoder and decoder
- BranchViT: Hybrid architecture combining ViT and ResNet

Author: [Your Name]
Date: [Date]
Version: 2.0
"""

from torchvision import models
import torch.nn.functional as F
import os
import timm
import torch
import torch.nn as nn
from torch.nn import init
from typing import Dict, List, Optional, Tuple, Union
import logging

from models.networks import ResnetGenerator, get_norm_layer, init_net

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print detailed model information

    Args:
        model: PyTorch model
        model_name: Model name
    """
    print(f"\n{'=' * 60}")
    print(f"  {model_name} Model Information")
    print(f"{'=' * 60}")

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    print(f"{'=' * 60}\n")


def load_checkpoint(model: nn.Module,
                    pretrained: bool,
                    pretrained_path: str,
                    model_name: str = "Model") -> nn.Module:
    """
    Load pretrained checkpoint

    Args:
        model: Model to load weights into
        pretrained: Whether to use pretrained weights
        pretrained_path: Path to pretrained weight file
        model_name: Model name for logging

    Returns:
        Model with loaded weights
    """
    print(f"\n{'=' * 50}")
    print(f"  {model_name} Weight Loading Status")
    print(f"{'=' * 50}")

    if not pretrained:
        print(f"❌ Pretrained weight loading disabled")
        print(f"📝 {model_name} will train from scratch")
        print(f"{'=' * 50}\n")
        return model

    if not os.path.isfile(pretrained_path):
        print(f"❌ Pretrained weight file not found: {pretrained_path}")
        print(f"📝 {model_name} will train from scratch")
        print(f"{'=' * 50}\n")
        return model

    try:
        print(f"🔄 Loading pretrained weights: {pretrained_path}")
        print(f"📁 File size: {os.path.getsize(pretrained_path) / 1024 / 1024:.2f} MB")

        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get('model', checkpoint)

        # Check weight dictionary
        print(f"📊 Pretrained weights contain {len(state_dict)} layers")

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Detailed loading results
        print(f"\n📈 Weight loading results:")
        print(f"  ✅ Successfully loaded layers: {len(state_dict) - len(missing_keys)}")
        print(f"  ⚠️  Missing layers: {len(missing_keys)}")
        print(f"  ❓ Unexpected layers: {len(unexpected_keys)}")

        if missing_keys:
            print(f"\n⚠️  Missing layers (will use random initialization):")
            for i, key in enumerate(missing_keys[:10]):  # Show only first 10
                print(f"     {i + 1:2d}. {key}")
            if len(missing_keys) > 10:
                print(f"     ... and {len(missing_keys) - 10} more missing layers")

        if unexpected_keys:
            print(f"\n❓ Unexpected layers (will be ignored):")
            for i, key in enumerate(unexpected_keys[:10]):  # Show only first 10
                print(f"     {i + 1:2d}. {key}")
            if len(unexpected_keys) > 10:
                print(f"     ... and {len(unexpected_keys) - 10} more unexpected layers")

        print(f"\n✅ Pretrained weights loaded successfully: {model_name}")
        print(f"{'=' * 50}\n")

    except Exception as e:
        print(f"❌ Error loading pretrained weights: {str(e)}")
        print(f"📝 {model_name} will train from scratch")
        print(f"{'=' * 50}\n")

    return model


def load_dino_checkpoint(model: nn.Module,
                         pretrained: bool,
                         pretrained_path: str,
                         model_name: str = "DINO Model") -> nn.Module:
    """
    Load DINO pretrained checkpoint with special key transformations

    Args:
        model: Model to load weights into
        pretrained: Whether to use pretrained weights
        pretrained_path: Path to pretrained weight file
        model_name: Model name for logging

    Returns:
        Model with loaded weights
    """
    print(f"\n{'=' * 50}")
    print(f"  {model_name} DINO Weight Loading Status")
    print(f"{'=' * 50}")

    if not pretrained:
        print(f"❌ DINO pretrained weight loading disabled")
        print(f"📝 {model_name} will train from scratch")
        print(f"{'=' * 50}\n")
        return model

    if not os.path.isfile(pretrained_path):
        print(f"❌ DINO pretrained weight file not found: {pretrained_path}")
        print(f"📝 {model_name} will train from scratch")
        print(f"{'=' * 50}\n")
        return model

    try:
        import re

        print(f"🔄 Loading DINO pretrained weights: {pretrained_path}")
        print(f"📁 File size: {os.path.getsize(pretrained_path) / 1024 / 1024:.2f} MB")

        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        print(f"🔧 Performing DINO weight key transformations...")
        new_state_dict = {}
        nan_count = 0

        for k, v in state_dict.items():
            # DINO-specific key transformations
            new_key = k.replace('teacher.backbone.blocks.', 'blocks.')
            new_key = re.sub(r'\.\d+\.(?=\d)', r'.', new_key)
            new_key = new_key.replace('student.backbone.', '')

            # Check for NaN values
            if torch.isnan(v).any():
                print(f"⚠️  Found NaN values in layer: {new_key}")
                # Reinitialize with Gaussian distribution
                v = torch.empty_like(v)
                init.normal_(v, mean=0, std=0.02)
                nan_count += 1
                print(f"🔧 Reinitialized layer: {new_key}")

            new_state_dict[new_key] = v

        if nan_count > 0:
            print(f"📊 Total reinitialized layers with NaN: {nan_count}")

        # Load transformed weights
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        # Detailed loading results
        print(f"\n📈 DINO weight loading results:")
        print(f"  ✅ Successfully loaded layers: {len(new_state_dict) - len(missing_keys)}")
        print(f"  ⚠️  Missing layers: {len(missing_keys)}")
        print(f"  ❓ Unexpected layers: {len(unexpected_keys)}")
        print(f"  🔧 Reinitialized layers: {nan_count}")

        if missing_keys:
            print(f"\n⚠️  Missing layers:")
            for i, key in enumerate(missing_keys[:5]):  # Show only first 5
                print(f"     {i + 1}. {key}")
            if len(missing_keys) > 5:
                print(f"     ... and {len(missing_keys) - 5} more missing layers")

        print(f"\n✅ DINO pretrained weights loaded successfully: {model_name}")
        print(f"{'=' * 50}\n")

    except Exception as e:
        print(f"❌ Error loading DINO pretrained weights: {str(e)}")
        print(f"📝 {model_name} will train from scratch")
        print(f"{'=' * 50}\n")

    return model


def define_Gv2(input_nc: int,
               output_nc: int,
               ngf: int,
               netG_nblocks: int,
               netG_branch: str,
               pretained_ImageNet: bool,
               pretrained_OwnResnet18: bool,
               pretrained_USFM: bool,
               pretrained_75per: bool,
               pretrained_Dino: bool,
               pretrained_dir: str,
               norm: str = 'batch',
               use_dropout: bool = False,
               init_type: str = 'normal',
               init_gain: float = 0.02,
               gpu_ids: List[int] = []) -> nn.Module:
    """
    Create generator network

    Args:
        input_nc: Number of input image channels
        output_nc: Number of output image channels
        ngf: Number of filters in the last conv layer
        netG_nblocks: Counts of Generator block (6, 9)
        netG_branch: Network architecture branch name ('resnet18_branch' | 'vit_base_branch')
        pretained_ImageNet: Whether to use ImageNet pretrained weights
        pretrained_OwnResnet18: Whether to use custom ResNet18 pretrained weights
        pretrained_USFM: Whether to use USFM pretrained weights
        pretrained_75per: Whether to use 75% pretrained weights
        pretrained_Dino: Whether to use DINO pretrained weights
        pretrained_dir: Pretrained weights directory
        norm: Normalization layer type
        use_dropout: Whether to use dropout
        init_type: Initialization method
        init_gain: Initialization gain
        gpu_ids: GPU device list

    Returns:
        Initialized generator network
    """
    print(f"\n{'=' * 60}")
    print(f"  Creating Generator Network")
    print(f"{'=' * 60}")
    print(f"📋 Network Configuration:")
    print(f"   Num of blocks: {netG_nblocks}")
    print(f"   Input/Output channels: {input_nc} -> {output_nc}")
    print(f"   Base filter count: {ngf}")
    print(f"   Normalization type: {norm}")
    print(f"   Use Dropout: {use_dropout}")
    print(f"   Initialization method: {init_type}")
    print(f"   Initialization gain: {init_gain}")
    print(f"   GPU devices: {gpu_ids if gpu_ids else 'CPU'}")

    # Pretrained options summary
    pretrain_options = {
        'ImageNet': pretained_ImageNet,
        'OwnResnet18': pretrained_OwnResnet18,
        'USFM': pretrained_USFM,
        '75per': pretrained_75per,
        'DINO': pretrained_Dino
    }
    active_pretrains = [k for k, v in pretrain_options.items() if v]
    print(f"   Active pretrained options: {active_pretrains if active_pretrains else ['None (train from scratch)']}")
    print(f"   Pretrained weights directory: {pretrained_dir}")
    print(f"{'=' * 60}")

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG_branch == 'resnet18_branch':
        print(f"🏗️  Building ResNet18 Branch Network...")
        net = BranchResNet(
            input_nc=input_nc,
            output_nc=output_nc,
            pretrained_OwnDataset=pretrained_OwnResnet18,
            pretrained_ImageNet=pretained_ImageNet,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=netG_nblocks,
        )
        print(f"✅ ResNet18 Branch Network construction completed")

    elif netG_branch == "vit_base_branch":
        print(f"🏗️  Building ViT Base Branch Network...")
        net = BranchViT(
            input_nc=input_nc,
            output_nc=output_nc,
            pretrained_ImageNet=pretained_ImageNet,
            pretrained_USFM=pretrained_USFM,
            pretrained_75per=pretrained_75per,
            pretrained_Dino=pretrained_Dino,
            pretrained_dirs=pretrained_dir,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=netG_nblocks,
        )
        print(f"✅ ViT Base Branch Network construction completed")

    else:
        available_models = ['resnet18_branch', 'vit_base_branch']
        raise NotImplementedError(f'❌ Unsupported generator branch model: {netG_branch}. Available models: {available_models}')

    # Print model information
    print_model_info(net, f"{netG_branch} Generator")

    print(f"🔧 Initializing network weights...")
    net = init_net(net, init_type, init_gain, gpu_ids)
    print(f"✅ Network initialization completed\n")

    return net


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 Encoder: ResNet-18 network without classification head

    Supports multiple pretrained weight loading options:
    - ImageNet pretrained weights
    - Custom dataset pretrained weights
    """

    def __init__(self,
                 pretrained_ImageNet: bool,
                 pretrained_OwnDataset: bool,
                 pretrained_dirs: str = 'weights/C2Fresnet18.pt'):
        """
        Initialize ResNet18 encoder

        Args:
            pretrained_ImageNet: Whether to use ImageNet pretrained weights
            pretrained_OwnDataset: Whether to use custom dataset pretrained weights
            pretrained_dirs: Custom pretrained weights path
        """
        super(ResNet18Encoder, self).__init__()

        print(f"\n🏗️  Initializing ResNet18 Encoder...")

        if pretrained_OwnDataset:
            print(f"📥 Loading custom ResNet18 weights: {pretrained_dirs}")
            resnet18 = models.resnet18(pretrained=False)

            if os.path.isfile(pretrained_dirs):
                try:
                    state_dict = torch.load(pretrained_dirs, map_location="cpu")
                    # Handle module name prefix
                    new_state_dict = {k.replace("module.backbone.", ""): v for k, v in state_dict.items()}
                    missing_keys, unexpected_keys = resnet18.load_state_dict(new_state_dict, strict=False)

                    print(f"✅ Custom weights loaded successfully")
                    print(f"   Missing layers: {len(missing_keys)}")
                    print(f"   Unexpected layers: {len(unexpected_keys)}")
                except Exception as e:
                    print(f"❌ Failed to load custom weights: {e}")
                    print(f"🔄 Switching to ImageNet weights")
                    pretrained_ImageNet = True
                    pretrained_OwnDataset = False
            else:
                print(f"❌ Custom weight file not found: {pretrained_dirs}")
                print(f"🔄 Switching to ImageNet weights")
                pretrained_ImageNet = True
                pretrained_OwnDataset = False

        if not pretrained_OwnDataset:
            if pretrained_ImageNet:
                print(f"📥 Loading ImageNet pretrained weights to ResNet18 encoder")
                resnet18 = models.resnet18(pretrained=True)
                print(f"✅ ImageNet weights loaded successfully")
            else:
                print(f"🎲 ResNet18 encoder will train from scratch")
                resnet18 = models.resnet18(pretrained=False)

        # Build encoder (remove fully connected layer)
        self.encoder = nn.Sequential(
            resnet18.conv1,  # 7x7 conv, 64
            resnet18.bn1,  # BatchNorm
            resnet18.relu,  # ReLU
            resnet18.maxpool,  # 3x3 max pooling
            resnet18.layer1,  # 64 channels
            resnet18.layer2,  # 128 channels
            resnet18.layer3,  # 256 channels
            resnet18.layer4,  # 512 channels
        )

        print(f"✅ ResNet18 encoder initialization completed")
        print(f"   Output feature dimension: 512 channels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Encoded features (B, 512, H/32, W/32)
        """
        return self.encoder(x)


class ResnetDecoder(nn.Module):
    """
    ResNet Symmetric Decoder: Symmetric decoding structure to ResNet-18 encoder
    Uses transposed convolutions for upsampling reconstruction
    """

    def __init__(self):
        """Initialize ResNet decoder"""
        super(ResnetDecoder, self).__init__()

        print(f"🏗️  Initializing ResNet Symmetric Decoder...")

        # Symmetric decoder structure using transposed convolutions for upsampling
        self.decoder = nn.Sequential(
            # First upsampling layer: 512 -> 256, 2x upsampling
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),

            # Second upsampling layer: 256 -> 256, 2x upsampling
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),

            # Third upsampling layer: 256 -> 256, 2x upsampling
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        print(f"✅ ResNet decoder initialization completed")
        print(f"   Input feature dimension: 512 channels")
        print(f"   Output feature dimension: 256 channels")
        print(f"   Upsampling factor: 8x")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Encoder output features (B, 512, H/32, W/32)

        Returns:
            Decoded features (B, 256, H/4, W/4)
        """
        return self.decoder(x)


class BranchResNet(ResnetGenerator):
    """
    ResNet Branch Network: Autoencoder combining ResNet-18 encoder and symmetric decoder

    Network Structure:
    1. ResNet18 encoder branch: Extract deep features
    2. CycleGAN encoder branch: Extract style features
    3. Feature fusion: Concatenate features from both branches
    4. CycleGAN decoder: Generate final output
    """

    def __init__(self,
                 input_nc: int,
                 output_nc: int,
                 pretrained_OwnDataset: bool,
                 pretrained_ImageNet: bool,
                 ngf: int = 64,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 use_dropout: bool = False,
                 n_blocks: int = 6,
                 padding_type: str = 'reflect'):
        """
        Initialize BranchResNet

        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            pretrained_OwnDataset: Whether to use custom dataset pretraining
            pretrained_ImageNet: Whether to use ImageNet pretraining
            ngf: Generator base filter count
            norm_layer: Normalization layer
            use_dropout: Whether to use dropout
            n_blocks: Number of ResNet blocks
            padding_type: Padding type
        """
        super(BranchResNet, self).__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
            padding_type=padding_type
        )

        print(f"\n🏗️  Initializing BranchResNet Network...")
        print(f"📋 Network Parameters:")
        print(f"   Input/Output channels: {input_nc} -> {output_nc}")
        print(f"   Base filter count: {ngf}")
        print(f"   ResNet block count: {n_blocks}")
        print(f"   Padding type: {padding_type}")

        # ResNet18 encoder branch
        self.encoder = ResNet18Encoder(
            pretrained_ImageNet=pretrained_ImageNet,
            pretrained_OwnDataset=pretrained_OwnDataset
        )

        # ResNet decoder branch
        self.decoder = ResnetDecoder()

        # CycleGAN backbone network
        print(f"🏗️  Initializing CycleGAN Backbone Network...")
        self.cycleGAN = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type
        )

        # Separate CycleGAN encoder and decoder
        self.cycleGAN_encoder = self.cycleGAN.model[:-9]  # Front encoding layers
        self.cycleGAN_decoder = self.cycleGAN.model[-9:]  # Back decoding layers

        # Feature fusion layer: 512(ResNet branch) + 256(CycleGAN branch) -> 256
        self.additional = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(256, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        print(f"✅ BranchResNet network initialization completed")
        print(f"📊 Network Architecture:")
        print(f"   ResNet18 branch: input -> 512 features -> 256 features")
        print(f"   CycleGAN branch: input -> 256 features")
        print(f"   Feature fusion: concat(256+256) -> 256")
        print(f"   Final output: 256 features -> output")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Generated image (B, C, H, W)
        """
        # ResNet18 branch: encode -> decode
        reparation = self.decoder(self.encoder(x))

        # CycleGAN branch: encode only
        cycleGAN_encoder_output = self.cycleGAN_encoder(x)

        # Feature fusion
        x = torch.cat((reparation, cycleGAN_encoder_output), dim=1)
        x = self.additional(x)

        # Final decoding
        x = self.cycleGAN_decoder(x)

        return x


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer Feature Extractor

    Supports multiple pretrained weights:
    - ImageNet pretraining
    - USFM pretraining
    - 75% data pretraining
    - DINO self-supervised pretraining
    """

    def __init__(self,
                 pretrained_ImageNet: bool,
                 pretrained_USFM: bool,
                 pretrained_75per: bool,
                 pretrained_Dino: bool,
                 pretrained_dirs: str):
        """
        Initialize ViT feature extractor

        Args:
            pretrained_ImageNet: Whether to use ImageNet pretraining
            pretrained_USFM: Whether to use USFM pretraining
            pretrained_75per: Whether to use 75% data pretraining
            pretrained_Dino: Whether to use DINO pretraining
            pretrained_dirs: Pretrained weights path
        """
        super().__init__()

        print(f"\n🏗️  Initializing ViT Base Feature Extractor...")

        # Load model based on pretraining options
        if pretrained_USFM:
            print(f"📥 Loading USFM pretrained ViT model")
            self.vit_b = timm.create_model('vit_base_patch16_224', pretrained=False)
            self.vit_b = load_checkpoint(self.vit_b, pretrained_USFM, pretrained_dirs, "USFM ViT")

        elif pretrained_75per:
            print(f"📥 Loading 75% data pretrained ViT model")
            self.vit_b = timm.create_model('vit_base_patch16_224', pretrained=False)
            self.vit_b = load_checkpoint(self.vit_b, pretrained_75per, pretrained_dirs, "75% ViT")

        elif pretrained_Dino:
            print(f"📥 Loading DINO pretrained ViT model")
            self.vit_b = timm.create_model('vit_base_patch16_224', pretrained=False)
            self.vit_b = load_dino_checkpoint(self.vit_b, pretrained_Dino, pretrained_dirs, "DINO ViT")

        else:
            if pretrained_ImageNet:
                print(f"📥 Loading ImageNet pretrained ViT model")
                self.vit_b = timm.create_model('vit_base_patch16_224', pretrained=True)
                print(f"✅ ImageNet ViT weights loaded successfully")
            else:
                print(f"🎲 ViT model will train from scratch")
                self.vit_b = timm.create_model('vit_base_patch16_224', pretrained=False)

        # Remove classification head, use identity mapping
        self.vit_b.head = nn.Identity()

        print(f"✅ ViT feature extractor initialization completed")
        print(f"📊 ViT Configuration:")
        print(f"   Model: vit_base_patch16_224")
        print(f"   Input resolution: 224x224")
        print(f"   Patch size: 16x16")
        print(f"   Output feature dimension: 768")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input image (B, C, H, W)

        Returns:
            ViT features (B, 768)
        """
        # Resize input to 224x224 (ViT standard input)
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Extract global features (CLS token features)
        outputs = self.vit_b(x)
        return outputs


class ViTDecoder(nn.Module):
    """
    ViT Decoder: Reconstruct spatial feature maps from ViT global features

    Decoding Process:
    1. Global features -> Patch-level features
    2. Upsample to reconstruct spatial structure
    """

    def __init__(self,
                 patch_size: int,
                 num_patches: int,
                 hidden_dim: int,
                 image_size: int):
        """
        Initialize ViT decoder

        Args:
            patch_size: Patch size
            num_patches: Total number of patches
            hidden_dim: Hidden layer dimension
            image_size: Image size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size

        print(f"🏗️  Initializing ViT Decoder...")
        print(f"📊 Decoder Configuration:")
        print(f"   Patch size: {patch_size}x{patch_size}")
        print(f"   Number of patches: {num_patches}")
        print(f"   Hidden dimension: {hidden_dim}")
        print(f"   Image size: {image_size}x{image_size}")

        # Reconstruct from global features to patch features
        self.re_embed = nn.Linear(hidden_dim, patch_size * patch_size * 3)

        # Upsampling module: reconstruct from patch level to full image
        self.upsample = nn.Sequential(
            # First upsampling layer
            nn.ConvTranspose2d(3, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),

            # Intermediate feature enhancement
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(128, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),

            # Second upsampling layer
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        print(f"✅ ViT decoder initialization completed")
        print(f"   Input: Global features (768-dim)")
        print(f"   Output: Spatial features (256 channels)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: ViT global features (B, 768)

        Returns:
            Reconstructed spatial features (B, 256, H, W)
        """
        # Reconstruct to patch space
        x = self.re_embed(x)  # (B, patch_size^2 * 3)
        x = x.view(-1, 3, self.patch_size, self.patch_size)

        # Upsample to reconstruct spatial structure
        reparation = self.upsample(x)
        return reparation


class BranchViT(ResnetGenerator):
    """
    ViT Branch Network: Hybrid architecture combining Vision Transformer and ResNet

    Network Structure:
    1. ViT feature extractor: Extract global semantic features
    2. ViT decoder: Reconstruct spatial features
    3. CycleGAN encoder: Extract local detail features
    4. Feature fusion: Combine global and local features
    5. CycleGAN decoder: Generate final output
    """

    def __init__(self,
                 input_nc: int,
                 output_nc: int,
                 pretrained_ImageNet: bool,
                 pretrained_USFM: bool,
                 pretrained_75per: bool,
                 pretrained_Dino: bool,
                 pretrained_dirs: str,
                 ngf: int = 64,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 use_dropout: bool = False,
                 n_blocks: int = 6,
                 padding_type: str = 'reflect',
                 image_size: int = 224,
                 patch_size: int = 16):
        """
        Initialize BranchViT network

        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            pretrained_ImageNet: Whether to use ImageNet pretraining
            pretrained_USFM: Whether to use USFM pretraining
            pretrained_75per: Whether to use 75% data pretraining
            pretrained_Dino: Whether to use DINO pretraining
            pretrained_dirs: Pretrained weights path
            ngf: Generator base filter count
            norm_layer: Normalization layer
            use_dropout: Whether to use dropout
            n_blocks: Number of ResNet blocks
            padding_type: Padding type
            image_size: Image size
            patch_size: Patch size
        """
        super(BranchViT, self).__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=n_blocks,
            padding_type=padding_type
        )

        print(f"\n🏗️  Initializing BranchViT Hybrid Network...")
        print(f"📋 Network Parameters:")
        print(f"   Input/Output channels: {input_nc} -> {output_nc}")
        print(f"   Base filter count: {ngf}")
        print(f"   ResNet block count: {n_blocks}")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Patch size: {patch_size}x{patch_size}")

        # ViT feature extractor
        self.encoder = ViTFeatureExtractor(
            pretrained_ImageNet=pretrained_ImageNet,
            pretrained_USFM=pretrained_USFM,
            pretrained_75per=pretrained_75per,
            pretrained_Dino=pretrained_Dino,
            pretrained_dirs=pretrained_dirs
        )

        # ViT decoder
        self.decoder = ViTDecoder(
            patch_size=patch_size,
            num_patches=(image_size // patch_size) ** 2,
            hidden_dim=768,  # ViT Base standard hidden dimension
            image_size=image_size,
        )

        # CycleGAN backbone network
        print(f"🏗️  Initializing CycleGAN Backbone Network...")
        self.cycleGAN = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, padding_type
        )

        # Separate CycleGAN encoder and decoder
        self.cycleGAN_encoder = self.cycleGAN.model[:-9]
        # print("cycleGAN_encoder:", self.cycleGAN_encoder)
        self.cycleGAN_decoder = self.cycleGAN.model[-9:]
        # print("cycleGAN_decoder:", self.cycleGAN_decoder)

        # Feature fusion layer: 512(ViT branch + CycleGAN branch) -> 256
        self.additional = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(256, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        print(f"✅ BranchViT hybrid network initialization completed")
        print(f"📊 Network Architecture:")
        print(f"   ViT branch: input -> 768 global features -> 256 spatial features")
        print(f"   CycleGAN branch: input -> 256 spatial features")
        print(f"   Feature fusion: concat(256+256) -> 256")
        print(f"   Final output: 256 features -> output")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input image (B, C, H, W)

        Returns:
            Generated image (B, C, H, W)
        """
        # ViT branch: global feature extraction -> spatial feature reconstruction
        reparation = self.decoder(self.encoder(x))

        # CycleGAN branch: local feature extraction
        cycleGAN_encoder_output = self.cycleGAN_encoder(x)

        # Feature fusion
        x = torch.cat((reparation, cycleGAN_encoder_output), dim=1)
        x = self.additional(x)

        # Final decoding
        x = self.cycleGAN_decoder(x)

        return x


def test_model():
    """Example function to test model functionality"""
    print(f"\n{'=' * 60}")
    print(f"  Model Testing Example")
    print(f"{'=' * 60}")

    # Create ViT branch model for testing
    model = BranchViT(
        input_nc=3,
        output_nc=3,
        pretrained_ImageNet=False,
        pretrained_USFM=False,
        pretrained_75per=False,
        pretrained_Dino=False,
        pretrained_dirs="",
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=False,
        n_blocks=9
    )

    # Simulate input
    input_image = torch.randn(2, 3, 256, 256)
    print(f"📥 Test input size: {input_image.shape}")

    # Forward pass
    with torch.no_grad():
        output_image = model(input_image)

    print(f"📤 Test output size: {output_image.shape}")
    print(f"✅ Model testing completed")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    test_model()
