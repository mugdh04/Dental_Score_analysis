"""Test DINOv2 loading with timm key remapping fix."""
import timm, torch, os

WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth'

def load_dinov2_backbone(model_name='vit_base_patch14_reg4_dinov2', device='cpu'):
    """Load DINOv2 with key remapping fix for timm 1.0.25."""
    model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='avg')
    
    # Get the pretrained weights URL
    cfg = model.pretrained_cfg
    url = cfg.get('url', WEIGHTS_URL)
    print(f'Downloading weights from: {url}')
    
    # Download pretrained state dict
    sd = torch.hub.load_state_dict_from_url(url, map_location=device)
    
    # Remap keys: norm.* -> fc_norm.*  (timm 1.0.25 compatibility fix)
    remapped = {}
    for k, v in sd.items():
        if k == 'norm.weight':
            remapped['fc_norm.weight'] = v
        elif k == 'norm.bias':
            remapped['fc_norm.bias'] = v
        elif k == 'mask_token':
            continue
        else:
            remapped[k] = v
    
    result = model.load_state_dict(remapped, strict=False)
    if result.missing_keys:
        print(f'  Missing keys: {result.missing_keys}')
    if result.unexpected_keys:
        print(f'  Unexpected keys: {result.unexpected_keys}')
    
    return model

if __name__ == '__main__':
    print('Loading DINOv2-base with register tokens...')
    m = load_dinov2_backbone()
    m.eval()
    print(f'Model loaded: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params')
    
    with torch.no_grad():
        out = m(torch.randn(1, 3, 224, 224))
    print(f'Output shape: {out.shape}')
    print('SUCCESS!')
