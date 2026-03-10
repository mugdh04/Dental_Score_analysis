"""
Loss functions for ordinal regression (CORAL approach).
Consistent Rank Logits (CORAL) for ordinal classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralOrdinalLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) ordinal regression loss.

    For a K-class ordinal problem, the model outputs K-1 logits.
    Each logit represents P(Y > k) for k = 0, ..., K-2.
    The target is a cumulative binary vector: [1,1,...,0,0]
    where the transition from 1→0 occurs at the true label.

    Loss = sum of binary cross-entropy for each threshold.
    """

    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        if class_weights is not None:
            self.register_buffer('class_weights', torch.FloatTensor(class_weights))
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K-1) raw logits for K-1 thresholds
            targets: (B, K-1) cumulative binary targets

        Returns:
            Scalar loss
        """
        # Binary cross-entropy with logits for each threshold
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Average across thresholds, then across batch
        loss = loss.mean(dim=1)  # (B,)

        if self.class_weights is not None:
            # Convert cumulative targets back to class labels for weighting
            # Class = sum of ones in cumulative vector
            class_labels = targets.sum(dim=1).long()
            class_labels = torch.clamp(class_labels, 0, len(self.class_weights) - 1)
            weights = self.class_weights[class_labels]
            loss = loss * weights

        return loss.mean()


class MultiTaskOrdinalLoss(nn.Module):
    """
    Combined loss for multi-task ordinal regression.
    Weighted sum of CORAL losses for MGI, OHI, and GEI.
    """

    def __init__(self, mgi_weights=None, ohi_weights=None, gei_weights=None,
                 task_weights=None):
        super().__init__()

        self.mgi_loss = CoralOrdinalLoss(5, mgi_weights)  # 5 classes (0-4)
        self.ohi_loss = CoralOrdinalLoss(4, ohi_weights)  # 4 classes (0-3)
        self.gei_loss = CoralOrdinalLoss(3, gei_weights)  # 3 classes (0-2)

        # Task weighting (default equal)
        if task_weights is None:
            task_weights = [1.0, 1.0, 1.0]
        self.task_weights = task_weights

    def forward(self, outputs, batch):
        """
        Args:
            outputs: dict from model forward with 'mgi', 'ohi', 'gei' logits
            batch: dict from dataset with 'mgi_target', 'ohi_target', 'gei_target'

        Returns:
            total_loss, dict of individual losses
        """
        mgi_loss = self.mgi_loss(outputs['mgi'], batch['mgi_target'])
        ohi_loss = self.ohi_loss(outputs['ohi'], batch['ohi_target'])
        gei_loss = self.gei_loss(outputs['gei'], batch['gei_target'])

        total_loss = (
            self.task_weights[0] * mgi_loss +
            self.task_weights[1] * ohi_loss +
            self.task_weights[2] * gei_loss
        )

        losses = {
            'total': total_loss.item(),
            'mgi': mgi_loss.item(),
            'ohi': ohi_loss.item(),
            'gei': gei_loss.item(),
        }

        return total_loss, losses
