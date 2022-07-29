def swin_base_patch244_window1677_sthv2():

    cfg = dict(
        name="swin_base_patch244_window1677_sthv2",
        embed_dim=128, 
        depths=[2, 2, 18, 2], 
        num_heads=[4, 8, 16, 32], 
        patch_size=(2,4,4), 
        window_size=(16,7,7), 
        drop_path_rate=0.4, 
        patch_norm=True,
        link= "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth"
    )
    return cfg

def swin_base_patch244_window877_kinetics400_1k():

    cfg = dict(
        name="swin_base_patch244_window877_kinetics400_1k",
        embed_dim=128, 
        depths=[2, 2, 18, 2], 
        num_heads=[4, 8, 16, 32], 
        patch_size=(2,4,4), 
        window_size=(8,7,7), 
        drop_path_rate=0.3, 
        patch_norm=True,
        link= "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth"
    )
    return cfg

def swin_base_patch244_window877_kinetics400_22k():

    cfg = dict(
        name="swin_base_patch244_window877_kinetics400_22k",
        embed_dim=128, 
        depths=[2, 2, 18, 2], 
        num_heads=[4, 8, 16, 32], 
        patch_size=(2,4,4), 
        window_size=(8,7,7), 
        drop_path_rate=0.2, 
        patch_norm=True ,
        link= "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth"
    )
    return cfg

def swin_base_patch244_window877_kinetics600_22k():

    cfg = dict(
        name="swin_base_patch244_window877_kinetics400_22k",
        embed_dim=128, 
        depths=[2, 2, 18, 2], 
        num_heads=[4, 8, 16, 32], 
        patch_size=(2,4,4), 
        window_size=(8,7,7), 
        drop_path_rate=0.2, 
        patch_norm=True,
        link = "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth"
    )
    return cfg

def swin_small_patch244_window877_kinetics400_1k():

    cfg = dict(
        name="swin_small_patch244_window877_kinetics400_1k",
        embed_dim=96, 
        depths=[2, 2, 18, 2], 
        num_heads=[3, 6, 12, 24], 
        patch_size=(2,4,4), 
        window_size=(8,7,7), 
        drop_path_rate=0.1, 
        patch_norm=True,
        link = "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_small_patch244_window877_kinetics400_1k.pth"
    )
    return cfg


def swin_tiny_patch244_window877_kinetics400_1k():

    cfg = dict(
        name="swin_tiny_patch244_window877_kinetics400_1k",
        embed_dim=96, 
        depths=[2, 2, 6, 2], 
        num_heads=[3, 6, 12, 24], 
        patch_size=(2,4,4), 
        window_size=(8,7,7), 
        drop_path_rate=0.1, 
        patch_norm=True,
        link = "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth"
    )
    return cfg


MODEL_MAP = {
    'swin_tiny_patch244_window877_kinetics400_1k'  : swin_tiny_patch244_window877_kinetics400_1k,
    'swin_small_patch244_window877_kinetics400_1k' : swin_small_patch244_window877_kinetics400_1k,
    'swin_base_patch244_window877_kinetics400_22k' : swin_base_patch244_window877_kinetics400_22k,
    'swin_base_patch244_window877_kinetics600_22k' : swin_base_patch244_window877_kinetics600_22k,
    'swin_base_patch244_window877_kinetics400_1k'  : swin_base_patch244_window877_kinetics400_1k ,
    'swin_base_patch244_window1677_sthv2'          : swin_base_patch244_window1677_sthv2
}