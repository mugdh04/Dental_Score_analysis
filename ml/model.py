"""
Multi-view, multi-task dental index prediction model.
Architecture: Configurable timm backbone (shared) with 3 ordinal regression heads.
Supports EfficientNet, EfficientNetV2, ConvNeXt-V2, and any timm model.

Input: 3 dental photographs (frontal, left lateral, right lateral)
Output: 3 ordinal scores (MGI 0-4, OHI 0-3, GEI 0-2)
"""

import torch
import torch.nn as nn
import timm


class MultiViewDentalModel(nn.Module):
    """
    Multi-view multi-task model for dental index prediction.

    Architecture:
        1. Shared timm backbone processes each view independently
        2. Features from all 3 views are concatenated (3 x feature_dim)
        3. Shared representation layers
        4. Three separate classification heads for MGI, OHI, GEI
        5. Each head outputs logits for ordinal regression (K-1 thresholds)
    """

    def __init__(
        self,
        backbone_name='efficientnet_b3',
        pretrained=True,
        mgi_classes=5,
        ohi_classes=4,
        gei_classes=3,
        dropout=0.4,
    ):
        super().__init__()

        self.mgi_classes = mgi_classes
        self.ohi_classes = ohi_classes
        self.gei_classes = gei_classes

        # Shared backbone (any timm model)
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,       # Remove classification head (features only)
            global_pool='avg',   # Global average pooling
        )
        feature_dim = self.backbone.num_features

        # Multi-view fusion: concatenate features from 3 views
        fused_dim = feature_dim * 3  # 5376

        # Shared representation layers after fusion
        self.fusion_layers = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
        )

        # Task-specific heads (ordinal regression: K-1 thresholds)
        self.mgi_head = self._make_head(512, mgi_classes - 1, dropout * 0.5)
        self.ohi_head = self._make_head(512, ohi_classes - 1, dropout * 0.5)
        self.gei_head = self._make_head(512, gei_classes - 1, dropout * 0.5)

    def _make_head(self, in_features, num_thresholds, dropout):
        """Create a classification head for ordinal regression."""
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_thresholds),
        )

    def extract_features(self, x):
        """Extract features from a single view using the backbone."""
        return self.backbone(x)

    def forward(self, frontal, left_lateral, right_lateral):
        """
        Forward pass with 3 views.

        Args:
            frontal: Tensor (B, 3, H, W)
            left_lateral: Tensor (B, 3, H, W)
            right_lateral: Tensor (B, 3, H, W)

        Returns:
            dict with keys 'mgi', 'ohi', 'gei' containing logits
        """
        # Extract features from each view using the shared backbone
        feat_f = self.extract_features(frontal)       # (B, 1792)
        feat_l = self.extract_features(left_lateral)   # (B, 1792)
        feat_r = self.extract_features(right_lateral)  # (B, 1792)

        # Concatenate multi-view features
        fused = torch.cat([feat_f, feat_l, feat_r], dim=1)  # (B, 5376)

        # Shared representation
        shared = self.fusion_layers(fused)  # (B, 512)

        # Task-specific predictions (ordinal regression logits)
        mgi_logits = self.mgi_head(shared)  # (B, 4) - 4 thresholds for 5 classes
        ohi_logits = self.ohi_head(shared)  # (B, 3) - 3 thresholds for 4 classes
        gei_logits = self.gei_head(shared)  # (B, 1) - 1 threshold for 2... wait no, 2 thresholds for 3 classes

        return {
            'mgi': mgi_logits,
            'ohi': ohi_logits,
            'gei': gei_logits,
            'features': shared,  # For Grad-CAM or analysis
        }

    def predict_scores(self, frontal, left_lateral, right_lateral):
        """
        Predict ordinal scores from images.

        Returns:
            dict with predicted class labels and confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(frontal, left_lateral, right_lateral)

        results = {}
        for key in ['mgi', 'ohi', 'gei']:
            logits = outputs[key]
            # Convert ordinal logits to probabilities via sigmoid
            probs = torch.sigmoid(logits)
            # Predicted class = number of thresholds exceeded (>0.5)
            predicted = (probs > 0.5).sum(dim=1).int()
            # Confidence = mean probability certainty
            confidence = 1.0 - torch.mean(torch.abs(probs - 0.5) * 2, dim=1)
            confidence = torch.clamp(confidence, 0.0, 1.0)

            results[key] = {
                'score': predicted,
                'confidence': 1.0 - confidence,  # Higher is more certain
                'probs': probs,
            }

        return results


def build_model(pretrained=True, device='cpu'):
    """Build and return the multi-view dental model."""
    model = MultiViewDentalModel(pretrained=pretrained)
    model = model.to(device)
    return model


def load_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint.
    
    Reads backbone_name from the checkpoint config if available,
    so the correct architecture is reconstructed automatically.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine backbone from saved config
    backbone_name = 'efficientnet_b3'  # fallback default
    if 'config' in checkpoint and 'backbone_name' in checkpoint['config']:
        backbone_name = checkpoint['config']['backbone_name']
    
    model = MultiViewDentalModel(backbone_name=backbone_name, pretrained=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model
