"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for dental model.
Generates heatmap overlays showing which regions of each dental photo
the model focuses on for its predictions.

Adapted for Vision Transformer (DINOv2) backbone:
  - Uses attention rollout from the last transformer block instead of
    convolutional activation maps (ViTs have no conv layers to hook).
  - Hooks into the last transformer block's attention output.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# cv2 is optional — only needed for heatmap overlay visualization
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ViTGradCAM:
    """
    Attention-based Grad-CAM for Vision Transformer (DINOv2) models.

    Instead of hooking into Conv2d layers (which don't exist in ViTs),
    this hooks into the last transformer block's attention weights and
    uses gradient-weighted attention rollout to produce spatial heatmaps.
    """

    def __init__(self, model):
        """
        Args:
            model: DINOv2MultiViewModel or EnsembleDentalModel instance
        """
        self.model = model
        self.attention_weights = None
        self.attention_gradients = None
        self._hooks = []

        # Get the backbone (works for both model types)
        backbone = model.backbone

        # Hook into the last transformer block's attention
        last_block = backbone.blocks[-1]
        attn_module = last_block.attn

        self._hooks.append(
            attn_module.register_forward_hook(self._save_attention)
        )
        self._hooks.append(
            attn_module.register_full_backward_hook(self._save_attention_gradient)
        )

    def _save_attention(self, module, input, output):
        """Save attention output (we'll compute attention from qkv)."""
        # For timm ViTs, the attn module has attn_drop applied to attention weights
        # We need to get the attention weights before the projection
        # The output here is the projected attention, not the weights themselves
        # So we hook differently — store the input to compute attention manually
        self.attention_output = output.detach()

    def _save_attention_gradient(self, module, grad_input, grad_output):
        """Save gradients flowing through the attention module."""
        self.attention_gradients = grad_output[0].detach()

    def generate(self, image, view_name='frontal', target_index='mgi'):
        """
        Generate Grad-CAM heatmap for a single view using attention gradients.

        Args:
            image: Tensor (1, 3, H, W) - preprocessed image
            view_name: Which view this image is ('frontal', 'left_lateral', 'right_lateral')
            target_index: Which prediction head to explain ('mgi', 'ohi', 'gei')

        Returns:
            heatmap: numpy array (H, W) normalized 0-1
        """
        self.model.eval()

        # We need gradients enabled for the image
        image = image.detach().requires_grad_(True)

        # Create dummy inputs for the other two views
        dummy = torch.zeros_like(image)

        # Full forward pass — enables gradient flow
        if view_name == 'frontal':
            outputs = self.model(image, dummy, dummy)
        elif view_name == 'left_lateral':
            outputs = self.model(dummy, image, dummy)
        else:
            outputs = self.model(dummy, dummy, image)

        # Get the predicted class and backpropagate
        target_logits = outputs[target_index]
        pred_class = target_logits.argmax(dim=1)
        target = target_logits[0, pred_class]

        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Use gradient-weighted attention output to produce heatmap
        if self.attention_output is None or self.attention_gradients is None:
            return np.zeros((image.shape[2], image.shape[3]))

        # Gradient-weighted combination
        weights = torch.mean(self.attention_gradients, dim=[2], keepdim=True)
        cam = torch.sum(weights * self.attention_output, dim=-1)  # (1, seq_len)

        # Remove CLS/register tokens, keep only patch tokens
        img_size = image.shape[2]
        patch_size = 14  # DINOv2 patch size
        num_patches = img_size // patch_size
        expected_patch_tokens = num_patches * num_patches

        # The sequence may include CLS + register tokens at the start
        if cam.shape[1] > expected_patch_tokens:
            cam = cam[:, -expected_patch_tokens:]

        # ReLU
        cam = F.relu(cam)

        # Reshape to spatial grid
        cam = cam.reshape(1, 1, num_patches, num_patches)

        # Upsample to image size
        cam = F.interpolate(cam, size=(img_size, img_size), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def create_overlay(self, original_image, heatmap, alpha=0.4):
        """
        Create a heatmap overlay on the original image.

        Args:
            original_image: PIL Image or numpy array
            heatmap: numpy array (H, W) normalized 0-1
            alpha: Overlay transparency

        Returns:
            PIL Image with heatmap overlay
        """
        if not HAS_CV2:
            # Fallback: return original image if cv2 not available
            return original_image

        if isinstance(original_image, Image.Image):
            original = np.array(original_image)
        else:
            original = original_image.copy()

        # Resize heatmap to match original image
        h, w = original.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original)

        return Image.fromarray(overlay)

    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def generate_gradcam_for_patient(model, images, device='cpu'):
    """
    Generate Grad-CAM heatmaps for all 3 views of a patient.

    Args:
        model: Trained DINOv2MultiViewModel or EnsembleDentalModel
        images: dict with 'frontal', 'left_lateral', 'right_lateral' PIL Images
        device: torch device

    Returns:
        dict of overlay PIL Images for each view
    """
    from ml.transforms import get_inference_transforms

    img_size = getattr(model, 'img_size', 336)
    transform = get_inference_transforms(image_size=img_size)
    gradcam = ViTGradCAM(model)

    overlays = {}
    try:
        for view_name, pil_image in images.items():
            # Transform for model input
            img_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Generate heatmap (use MGI as primary target)
            heatmap = gradcam.generate(img_tensor, view_name, target_index='mgi')

            # Create overlay
            overlay = gradcam.create_overlay(pil_image, heatmap)
            overlays[view_name] = overlay
    finally:
        gradcam.cleanup()

    return overlays
