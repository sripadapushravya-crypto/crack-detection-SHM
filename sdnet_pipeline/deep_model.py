"""ResNet-18 model definition and Grad-CAM explainability, matching Chapter 3
§3.4 (fine-tuned ResNet-18, sigmoid output) and §3.10 (Grad-CAM on the final
convolutional block) of the thesis.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """
    ImageNet-pretrained ResNet-18 with the final fully-connected layer replaced
    by a single-logit head. Sigmoid is applied outside the model (in the loss
    and at inference time) rather than baked into the forward pass, which is
    the standard pattern for use with BCEWithLogitsLoss.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


class GradCAM:
    """
    Grad-CAM over the last convolutional block of ResNet-18 (layer4), matching
    the thesis's description of extracting feature maps A^k from "the final
    forward pass of the ResNet-18 layer stack" and pooling their gradients to
    obtain channel weights alpha_k (Selvaraju et al., 2017).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        input_tensor: a single-image batch, shape (1, 3, H, W), requires_grad
        not required on the input itself — gradients are captured on the
        target layer's output via the backward hook.

        Returns a (H, W) numpy array in [0, 1], upsampled to the input's
        spatial size, ready to be blended as a heatmap.
        """
        self.model.zero_grad(set_to_none=True)
        logit = self.model(input_tensor)
        logit.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire; check target layer.")

        # alpha_k = global average pool of gradients over spatial dims (Eq. in §3.10)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam
