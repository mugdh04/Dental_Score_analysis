"""Test which pretrained models load correctly with timm 1.0.25."""
import timm
import torch
import sys

def test_model(name, expected_dim):
    try:
        print(f'Testing {name}...', end=' ', flush=True)
        m = timm.create_model(name, pretrained=True, num_classes=0, global_pool='avg')
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 3, 224, 224))
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f'OK! Shape: {out.shape}, Params: {params:.1f}M')
        return True
    except Exception as e:
        print(f'FAILED: {e}')
        return False

# Test models from smallest to largest
models = [
    ('convnext_tiny', 768),
    ('convnext_small', 768),
    ('swin_tiny_patch4_window7_224', 768),
]

for name, dim in models:
    test_model(name, dim)

print('\nDone.')
