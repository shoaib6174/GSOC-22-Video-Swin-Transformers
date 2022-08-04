import tensorflow as tf
import torch
from VideoSwinTransformer import model_configs , SwinTransformer3D_pt
from collections import OrderedDict

tf_model = tf.keras.models.load_model('swin_tiny_patch244_window877_kinetics400_1k_tf.pth')


cfg_method = model_configs.MODEL_MAP["swin_tiny_patch244_window877_kinetics400_1k"]
cfg = cfg_method()

name = cfg["name"]
link = cfg['link']
del cfg["name"]
del cfg['link']

pt_model = SwinTransformer3D_pt(**cfg)

checkpoint = torch.load(f'{name}.pth')


new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    if 'backbone' in k:
        name = k[9:]
        new_state_dict[name] = v 

pt_model.load_state_dict(new_state_dict) 


print(tf_model.summary())

x = tf.random.normal((1,32, 224,224, 3), dtype='float64')

y = tf_model(x)
z = pt_model(x)

print(z.shape, y.shape)