"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for dental model.
Generates heatmap overlays showing which regions of each dental photo
the model focuses on for its predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    """
    Grad-CAM for multi-view dental model.
    Computes activation maps for the last convolutional layer of the backbone.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: MultiViewDentalModel instance
            target_layer: Target conv layer for Grad-CAM (default: last conv in backbone)
        """
        self.model = model
        self.gradients = None
        self.activations = None

        # Get the last convolutional layer of EfficientNet backbone
        if target_layer is None:
            # EfficientNet-B4's last conv layer before global pooling
            self.target_layer = self._find_last_conv(model.backbone)
        else:
            self.target_layer = target_layer

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _find_last_conv(self, backbone):
        """Find the last convolutional layer in the backbone."""
        last_conv = None
        for module in backbone.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, view_name='frontal', target_index='mgi'):
        """
        Generate Grad-CAM heatmap for a single view.

        Args:
            image: Tensor (1, 3, H, W) - preprocessed image
            view_name: Which view this image is ('frontal', 'left_lateral', 'right_lateral')
            target_index: Which prediction head to explain ('mgi', 'ohi', 'gei')

        Returns:
            heatmap: numpy array (H, W) normalized 0-1
        """
        self.model.eval()
        image.requires_grad_(True)

        # Forward pass through backbone only for this view
        features = self.model.backbone(image)

        # We need to track gradients through the specific head
        # Create dummy inputs for other views
        dummy = torch.zeros_like(image)

        # Full forward pass
        if view_name == 'frontal':
            outputs = self.model(image, dummy, dummy)
        elif view_name == 'left_lateral':
            outputs = self.model(dummy, image, dummy)
        else:
            outputs = self.model(dummy, dummy, image)

        # Get the target logits and backpropagate
        target_logits = outputs[target_index]

        # Use the sum of logits as the target for Grad-CAM
        target = target_logits.sum()

        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            return np.zeros((image.shape[2], image.shape[3]))

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Resize to input image size
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))

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


def generate_gradcam_for_patient(model, images, device='cpu'):
    """
    Generate Grad-CAM heatmaps for all 3 views of a patient.

    Args:
        model: Trained MultiViewDentalModel
        images: dict with 'frontal', 'left_lateral', 'right_lateral' PIL Images
        device: torch device

    Returns:
        dict of overlay PIL Images for each view
    """
    from ml.transforms import get_inference_transforms

    transform = get_inference_transforms()
    gradcam = GradCAM(model)

    overlays = {}
    for view_name, pil_image in images.items():
        # Transform for model input
        img_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Generate heatmap (use MGI as primary target)
        heatmap = gradcam.generate(img_tensor, view_name, target_index='mgi')

        # Create overlay
        overlay = gradcam.create_overlay(pil_image, heatmap)
        overlays[view_name] = overlay

    return overlays
