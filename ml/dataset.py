"""
Dataset module for loading thesis dental photographs and labels.
Handles multi-view (frontal, left lateral, right lateral) image loading
with matching MGI/OHI/GEI labels from CSV.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch


class DentalDataset(Dataset):
    """
    Multi-view dental dataset.
    Each sample contains 3 images (frontal, left_lateral, right_lateral)
    and 3 ordinal labels (MGI, OHI, GEI).
    """

    # Score ranges for ordinal regression
    MGI_CLASSES = 5   # 0, 1, 2, 3, 4
    OHI_CLASSES = 4   # 0, 1, 2, 3
    GEI_CLASSES = 3   # 0, 1, 2

    def __init__(self, data_dir, transform=None, patient_ids=None):
        """
        Args:
            data_dir: Path to Thesis_Data directory containing CSV and image folders.
            transform: torchvision transforms to apply to images.
            patient_ids: Optional list of patient IDs to include (for train/val split).
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load and parse CSV
        csv_path = self.data_dir / 'Thesis_Results.csv'
        self.df = pd.read_csv(csv_path)

        # Parse scores from the combined string column
        self.samples = []
        score_pattern = re.compile(r'MGI[- ]*(\d+).*?OHI[- ]*(\d+).*?GEI[- ]*(\d+)', re.IGNORECASE)

        for _, row in self.df.iterrows():
            sl_no = int(row.iloc[0])  # Sl No. column

            # Skip if patient_ids specified and this isn't in it
            if patient_ids is not None and sl_no not in patient_ids:
                continue

            # Parse the scores string
            score_str = str(row.iloc[2])  # Third column has scores
            match = score_pattern.search(score_str)
            if not match:
                continue

            mgi = int(match.group(1))
            ohi = int(match.group(2))
            gei = int(match.group(3))

            # Find image files for this patient
            frontal = self._find_image('Frontal', f'F{sl_no}')
            left = self._find_image('Left_Lateral', f'L{sl_no}')
            right = self._find_image('Right_Lateral', f'R{sl_no}')

            # Only include if ALL three views exist
            if frontal and left and right:
                self.samples.append({
                    'sl_no': sl_no,
                    'frontal': frontal,
                    'left_lateral': left,
                    'right_lateral': right,
                    'mgi': mgi,
                    'ohi': ohi,
                    'gei': gei,
                })

        print(f"Loaded {len(self.samples)} complete samples from {len(self.df)} entries")

    def _find_image(self, subfolder, prefix):
        """Find an image file matching the prefix in the subfolder."""
        photo_dir = self.data_dir / 'Thesis_Photographs' / subfolder
        if not photo_dir.exists():
            return None

        # Try common extensions
        for ext in ['.JPG', '.jpg', '.png', '.jpeg', '.JPEG', '.PNG']:
            path = photo_dir / f"{prefix}{ext}"
            if path.exists():
                return path

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        frontal = Image.open(sample['frontal']).convert('RGB')
        left = Image.open(sample['left_lateral']).convert('RGB')
        right = Image.open(sample['right_lateral']).convert('RGB')

        # Apply transforms
        if self.transform:
            frontal = self.transform(frontal)
            left = self.transform(left)
            right = self.transform(right)

        # Convert ordinal labels to cumulative binary targets for CORAL loss
        mgi_target = self._ordinal_to_binary(sample['mgi'], self.MGI_CLASSES)
        ohi_target = self._ordinal_to_binary(sample['ohi'], self.OHI_CLASSES)
        gei_target = self._ordinal_to_binary(sample['gei'], self.GEI_CLASSES)

        return {
            'frontal': frontal,
            'left_lateral': left,
            'right_lateral': right,
            'mgi_label': sample['mgi'],
            'ohi_label': sample['ohi'],
            'gei_label': sample['gei'],
            'mgi_target': torch.FloatTensor(mgi_target),
            'ohi_target': torch.FloatTensor(ohi_target),
            'gei_target': torch.FloatTensor(gei_target),
        }

    @staticmethod
    def _ordinal_to_binary(score, num_classes):
        """
        Convert ordinal label to cumulative binary vector.
        E.g., score=2 with 5 classes -> [1, 1, 0, 0] (4 thresholds)
        """
        return [1.0 if i < score else 0.0 for i in range(num_classes - 1)]


def get_patient_ids(data_dir):
    """Get all valid patient IDs that have complete image sets."""
    dataset = DentalDataset(data_dir)
    return [s['sl_no'] for s in dataset.samples]


def get_class_weights(data_dir):
    """Compute class weights for imbalanced dataset."""
    dataset = DentalDataset(data_dir)

    mgi_counts = np.zeros(DentalDataset.MGI_CLASSES)
    ohi_counts = np.zeros(DentalDataset.OHI_CLASSES)
    gei_counts = np.zeros(DentalDataset.GEI_CLASSES)

    for s in dataset.samples:
        mgi_counts[s['mgi']] += 1
        ohi_counts[s['ohi']] += 1
        gei_counts[s['gei']] += 1

    def compute_weights(counts):
        total = counts.sum()
        weights = total / (len(counts) * counts + 1e-6)
        return weights / weights.sum() * len(counts)

    return {
        'mgi': compute_weights(mgi_counts),
        'ohi': compute_weights(ohi_counts),
        'gei': compute_weights(gei_counts),
    }
