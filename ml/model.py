"""
Multi-view, multi-task dental index prediction model.

Architecture: DINOv2-base frozen backbone + lightweight MLP classifier.
DINOv2 (ViT-B/14) was self-supervised on 142M images and produces rich
visual features that transfer extremely well even with very few samples.

Strategy:
  1. DINOv2 backbone is FROZEN — no risk of overfitting
  2. Only a small MLP classifier head is trained (~300K params)
  3. K-fold cross-validation with ensemble averaging at inference
  4. Standard classification with class weights for imbalance handling

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
    """Lightweight MLP classifier on top of frozen DINOv2 features."""

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
    Full model: frozen DINOv2 backbone + trainable classifier head.
    Used at inference time in the Django app.
    """

    BACKBONE_NAME = 'vit_base_patch14_reg4_dinov2'
    FEATURE_DIM = 768  # per-view feature dim from DINOv2-base

    def __init__(self, dropout=0.4, pretrained_backbone=True):
        super().__init__()

        # Frozen DINOv2 backbone (with register tokens)
        self.backbone = _load_dinov2(self.BACKBONE_NAME, pretrained_backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Trainable classifier
        fused_dim = self.FEATURE_DIM * 3  # 768 * 3 = 2304
        self.classifier = DentalClassifierHead(
            input_dim=fused_dim, dropout=dropout)

    def extract_features(self, frontal, left_lateral, right_lateral):
        """Extract and concatenate DINOv2 features from 3 views."""
        with torch.no_grad():
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
        """Override: backbone always stays in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self


class EnsembleDentalModel(nn.Module):
    """
    Ensemble of K-fold classifier heads on a single shared DINOv2 backbone.
    At inference, averages softmax probabilities from all fold heads.
    """

    BACKBONE_NAME = 'vit_base_patch14_reg4_dinov2'
    FEATURE_DIM = 768

    def __init__(self, fold_head_paths, device='cpu'):
        super().__init__()
        self.device = device

        # Single shared frozen backbone (with register tokens)
        self.backbone = _load_dinov2(self.BACKBONE_NAME, pretrained=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Load all fold heads
        fused_dim = self.FEATURE_DIM * 3
        self.heads = nn.ModuleList()
        for path in fold_head_paths:
            head = DentalClassifierHead(input_dim=fused_dim)
            state = torch.load(path, map_location=device, weights_only=False)
            if 'head_state_dict' in state:
                head.load_state_dict(state['head_state_dict'])
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


def build_model(pretrained=True, device='cpu'):
    """Build the DINOv2 multi-view model."""
    model = DINOv2MultiViewModel(pretrained_backbone=pretrained)
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
        model = EnsembleDentalModel(fold_files, device=device)
        return model.to(device)

    # Single model fallback
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DINOv2MultiViewModel(pretrained_backbone=True)
    if 'head_state_dict' in ckpt:
        model.classifier.load_state_dict(ckpt['head_state_dict'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(device)
