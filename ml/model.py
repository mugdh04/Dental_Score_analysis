"""
Multi-view, multi-task dental index prediction model.

Architecture: DINOv2-base backbone (partially fine-tuned) + MLP classifier.
DINOv2 (ViT-B/14) was self-supervised on 142M images and produces rich
visual features that transfer extremely well even with very few samples.

Strategy:
  1. DINOv2 backbone: early layers frozen, last N blocks fine-tuned
  2. Differential learning rates: backbone 10-50x lower than head
  3. K-fold cross-validation with ensemble averaging at inference
  4. Weighted classification loss for class imbalance

Input: 3 dental photographs (frontal, left lateral, right lateral)
Output: 3 classification scores (MGI 0-4, OHI 0-3, GEI 0-2)
"""

import torch
import torch.nn as nn
import timm


def _interpolate_pos_embed(checkpoint_pos_embed, model_pos_embed):
    """Interpolate pos_embed from pretrained resolution to model's resolution."""
    tgt_len = model_pos_embed.shape[1]
    src_len = checkpoint_pos_embed.shape[1]
    dim = checkpoint_pos_embed.shape[2]
    if src_len == tgt_len:
        return checkpoint_pos_embed
    # Strip CLS token position (DINOv2 checkpoint = [1, 1+patches, dim])
    patch_pos = checkpoint_pos_embed[:, 1:, :]
    src_npatch = patch_pos.shape[1]
    if src_npatch == tgt_len:
        return patch_pos
    src_size = int(src_npatch ** 0.5)
    tgt_size = int(tgt_len ** 0.5)
    patch_pos = patch_pos.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2).float()
    patch_pos = torch.nn.functional.interpolate(
        patch_pos, size=(tgt_size, tgt_size), mode='bicubic', align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, dim)
    return patch_pos


def _load_dinov2(model_name, pretrained=True, img_size=224):
    """Load DINOv2 backbone with key/shape fixes for timm 1.0.25."""
    model = timm.create_model(model_name, pretrained=False, num_classes=0,
                               global_pool='avg', img_size=img_size)
    if pretrained:
        cfg = model.pretrained_cfg
        url = cfg.get('url', '')
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=False)
        model_sd = model.state_dict()
        remapped = {}
        for k, v in sd.items():
            target_key = k
            if k == 'norm.weight':
                target_key = 'fc_norm.weight'
            elif k == 'norm.bias':
                target_key = 'fc_norm.bias'
            elif k == 'mask_token':
                continue
            if target_key not in model_sd:
                continue
            if 'pos_embed' in target_key and v.shape != model_sd[target_key].shape:
                v = _interpolate_pos_embed(v, model_sd[target_key])
            if v.shape != model_sd[target_key].shape:
                continue
            remapped[target_key] = v
        model.load_state_dict(remapped, strict=False)
    return model


class DentalClassifierHead(nn.Module):
    """MLP classifier on top of DINOv2 features."""

    def __init__(self, input_dim=2304, hidden_dim=512, mgi_classes=5,
                 ohi_classes=4, gei_classes=3, dropout=0.4):
        super().__init__()
        self.mgi_classes = mgi_classes
        self.ohi_classes = ohi_classes
        self.gei_classes = gei_classes

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
        )
        self.mgi_head = nn.Linear(256, mgi_classes)
        self.ohi_head = nn.Linear(256, ohi_classes)
        self.gei_head = nn.Linear(256, gei_classes)

    def forward(self, features):
        """features: (B, 2304) concatenated DINOv2 features from 3 views."""
        h = self.shared(features)
        return {
            'mgi': self.mgi_head(h),
            'ohi': self.ohi_head(h),
            'gei': self.gei_head(h),
        }


class DINOv2MultiViewModel(nn.Module):
    """
    Full model: DINOv2 backbone (partially fine-tuned) + trainable classifier head.
    Supports freezing early backbone layers and fine-tuning the last N transformer blocks.
    """

    BACKBONE_NAME = 'vit_base_patch14_reg4_dinov2'
    FEATURE_DIM = 768  # per-view feature dim from DINOv2-base
    NUM_BLOCKS = 12     # DINOv2-base has 12 transformer blocks

    def __init__(self, dropout=0.4, pretrained_backbone=True, img_size=224,
                 unfreeze_blocks=0):
        super().__init__()
        self.img_size = img_size
        self.unfreeze_blocks = unfreeze_blocks

        # DINOv2 backbone (with register tokens)
        self.backbone = _load_dinov2(self.BACKBONE_NAME, pretrained_backbone,
                                      img_size=img_size)
        self._configure_backbone_freezing(unfreeze_blocks)

        # Trainable classifier
        fused_dim = self.FEATURE_DIM * 3  # 768 * 3 = 2304
        self.classifier = DentalClassifierHead(
            input_dim=fused_dim, dropout=dropout)

    def _configure_backbone_freezing(self, unfreeze_blocks):
        """Freeze all backbone params, then unfreeze the last N transformer blocks."""
        self.unfreeze_blocks = unfreeze_blocks

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        if unfreeze_blocks <= 0:
            return

        # Unfreeze fc_norm (final layer norm)
        for name, p in self.backbone.named_parameters():
            if 'fc_norm' in name:
                p.requires_grad = True

        # Unfreeze the last N blocks
        total_blocks = self.NUM_BLOCKS
        start_unfreeze = total_blocks - unfreeze_blocks
        for name, p in self.backbone.named_parameters():
            if 'blocks.' in name:
                block_idx = int(name.split('blocks.')[1].split('.')[0])
                if block_idx >= start_unfreeze:
                    p.requires_grad = True

    def get_param_groups(self, backbone_lr, head_lr, weight_decay=1e-2):
        """Get parameter groups with differential learning rates."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.classifier.parameters())

        groups = []
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': backbone_lr,
                          'weight_decay': weight_decay})
        groups.append({'params': head_params, 'lr': head_lr,
                      'weight_decay': weight_decay})
        return groups

    def extract_features(self, frontal, left_lateral, right_lateral):
        """Extract and concatenate DINOv2 features from 3 views."""
        feat_f = self.backbone(frontal)
        feat_l = self.backbone(left_lateral)
        feat_r = self.backbone(right_lateral)
        return torch.cat([feat_f, feat_l, feat_r], dim=1)

    def forward(self, frontal, left_lateral, right_lateral):
        features = self.extract_features(frontal, left_lateral, right_lateral)
        return self.classifier(features)

    def predict_scores(self, frontal, left_lateral, right_lateral):
        """Predict scores with softmax confidence (0-100%)."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(frontal, left_lateral, right_lateral)

        results = {}
        for key in ['mgi', 'ohi', 'gei']:
            probs = torch.softmax(outputs[key], dim=1)
            predicted = probs.argmax(dim=1).int()
            confidence = probs.max(dim=1).values * 100.0
            results[key] = {
                'score': predicted,
                'confidence': confidence,
                'probs': probs,
            }
        return results

    def train(self, mode=True):
        """Override: frozen backbone layers stay in eval mode; unfrozen layers can train."""
        super().train(mode)
        if self.unfreeze_blocks <= 0:
            self.backbone.eval()
        else:
            # Keep frozen layers in eval, unfrozen in train
            self.backbone.eval()
            start_unfreeze = self.NUM_BLOCKS - self.unfreeze_blocks
            for name, module in self.backbone.named_modules():
                if 'blocks.' in name:
                    parts = name.split('blocks.')[1].split('.')
                    if parts[0].isdigit():
                        block_idx = int(parts[0])
                        if block_idx >= start_unfreeze and mode:
                            module.train()
            # fc_norm in train mode
            if hasattr(self.backbone, 'fc_norm'):
                self.backbone.fc_norm.train(mode)
        return self


class EnsembleDentalModel(nn.Module):
    """
    Ensemble of K-fold full models on a single shared DINOv2 backbone.
    At inference, averages softmax probabilities from all fold heads.
    """

    BACKBONE_NAME = 'vit_base_patch14_reg4_dinov2'
    FEATURE_DIM = 768

    def __init__(self, fold_paths, device='cpu', img_size=224):
        super().__init__()
        self.device = device

        # Single shared backbone
        self.backbone = _load_dinov2(self.BACKBONE_NAME, pretrained=True,
                                      img_size=img_size)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Load all fold heads
        fused_dim = self.FEATURE_DIM * 3
        self.heads = nn.ModuleList()
        for path in fold_paths:
            head = DentalClassifierHead(input_dim=fused_dim)
            state = torch.load(path, map_location=device, weights_only=False)
            if 'head_state_dict' in state:
                head.load_state_dict(state['head_state_dict'])
            elif 'model_state_dict' in state:
                # Full model checkpoint — extract classifier weights
                full_sd = state['model_state_dict']
                head_sd = {}
                for k, v in full_sd.items():
                    if k.startswith('classifier.'):
                        head_sd[k.replace('classifier.', '')] = v
                if head_sd:
                    head.load_state_dict(head_sd)
                # Also load fine-tuned backbone weights from first fold
                if len(self.heads) == 0:
                    backbone_sd = {}
                    for k, v in full_sd.items():
                        if k.startswith('backbone.'):
                            backbone_sd[k.replace('backbone.', '')] = v
                    if backbone_sd:
                        self.backbone.load_state_dict(backbone_sd, strict=False)
            else:
                head.load_state_dict(state)
            head.eval()
            self.heads.append(head)

    def predict_scores(self, frontal, left_lateral, right_lateral):
        """Ensemble prediction: average softmax from all fold heads."""
        self.eval()
        with torch.no_grad():
            feat_f = self.backbone(frontal)
            feat_l = self.backbone(left_lateral)
            feat_r = self.backbone(right_lateral)
            features = torch.cat([feat_f, feat_l, feat_r], dim=1)

        results = {}
        for key in ['mgi', 'ohi', 'gei']:
            all_probs = []
            for head in self.heads:
                with torch.no_grad():
                    out = head(features)
                all_probs.append(torch.softmax(out[key], dim=1))

            avg_probs = torch.stack(all_probs).mean(dim=0)
            predicted = avg_probs.argmax(dim=1).int()
            confidence = avg_probs.max(dim=1).values * 100.0
            results[key] = {
                'score': predicted,
                'confidence': confidence,
                'probs': avg_probs,
            }
        return results


def build_model(pretrained=True, device='cpu', img_size=224, unfreeze_blocks=0):
    """Build the DINOv2 multi-view model."""
    model = DINOv2MultiViewModel(pretrained_backbone=pretrained,
                                  img_size=img_size,
                                  unfreeze_blocks=unfreeze_blocks)
    return model.to(device)


def load_model(checkpoint_path, device='cpu'):
    """
    Load a trained model from checkpoint.
    Supports both single-fold and ensemble checkpoints.
    """
    import glob
    import os

    checkpoint_path = str(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Check for ensemble (multiple fold files)
    fold_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'fold_*_head.pth')))

    if fold_files:
        # Detect img_size from checkpoint
        sample_ckpt = torch.load(fold_files[0], map_location=device, weights_only=False)
        img_size = sample_ckpt.get('config', {}).get('image_size', 224)
        model = EnsembleDentalModel(fold_files, device=device, img_size=img_size)
        return model.to(device)

    # Single model fallback
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    img_size = ckpt.get('config', {}).get('image_size', 224)
    model = DINOv2MultiViewModel(pretrained_backbone=True, img_size=img_size)

    if 'head_state_dict' in ckpt:
        model.classifier.load_state_dict(ckpt['head_state_dict'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(device)
