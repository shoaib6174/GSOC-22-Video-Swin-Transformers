import json
import argparse
import os
import sys

i = 1
model_layers = {}


import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
from collections import OrderedDict

from VideoSwinTransformer import SwinTransformer3D_pt
from VideoSwinTransformer import SwinTransformer3D
from VideoSwinTransformer import *


from VideoSwinTransformer import SwinTransformer3D, model_configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the PyTorch pre-trained Swin weights to TensorFlow."
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="swin_tiny_patch4_window7_224",
        type=str,
        choices=model_configs.MODEL_MAP.keys(),
        help="Name of the Swin model variant.",
    )

    return parser.parse_args()

def conv_transpose(w):
    return w.transpose(2,3,4,1, 0)
    
def modify_tf_block( tf_component, pt_weight,  pt_bias = None, is_attn=False, wn=None, bn=None):
   
    global i
    in_shape = pt_weight.shape
    # print(tf_component, pt_weight.shape)

    if isinstance(tf_component, tf.keras.layers.Conv3D) :
      pt_weight = conv_transpose(pt_weight)

    if isinstance(tf_component, tf.keras.layers.Dense) and not is_attn:
      pt_weight =pt_weight.transpose()

    if isinstance(tf_component, (tf.keras.layers.Dense, tf.keras.layers.Conv3D)):
        tf_component.kernel.assign(tf.Variable(pt_weight))
        #print(i)
#        i += 1
        model_layers[wn] = tf_component.kernel.name
        if pt_bias is not None:
            tf_component.bias.assign(tf.Variable(pt_bias))
            # print(i)
            # i += 1
            model_layers[bn] = tf_component.bias.name
        #print("dense/conv3d")
    elif isinstance(tf_component, tf.keras.layers.LayerNormalization):

        tf_component.gamma.assign(tf.Variable(pt_weight))
        #print(i)
#        i += 1
        model_layers[wn] = tf_component.gamma.name
        tf_component.beta.assign(tf.Variable(pt_bias))
        #print(i)
#        i += 1
        model_layers[bn] = tf_component.beta.name
        #print("layer norm")
    elif isinstance(tf_component, (tf.Variable)):
        # For regular variables (tf.Variable).
        tf_component.assign(tf.Variable(pt_weight))
        #print(i)
#        i += 1
        model_layers[wn] = tf_component.name
        #print("variable")
    else:
        #print("else")
        # print("else",i)
        # i += 1
        model_layers[wn] = tf_component.name
        return tf.convert_to_tensor(pt_weight)
        
        

    return tf_component


def modify_swin_blocks(np_state_dict, pt_weights_prefix, tf_block):
  # PatchMerging
  global i
  for layer in tf_block:
    if isinstance(layer, PatchMerging):
      patch_merging_idx = f"{pt_weights_prefix}.downsample"
  
      layer.reduction = modify_tf_block( layer.reduction,
                          np_state_dict[f"{patch_merging_idx}.reduction.weight"] , wn = f"{patch_merging_idx}.reduction.weight")
      
      layer.norm = modify_tf_block( layer.norm,
                        np_state_dict[f"{patch_merging_idx}.norm.weight"],
                        np_state_dict[f"{patch_merging_idx}.norm.bias"],
                        
                        wn = f"{patch_merging_idx}.norm.weight",
                        bn = f"{patch_merging_idx}.norm.bias"
                        )
      
  # Swin Layers
      # Swin layers.
  common_prefix = f"{pt_weights_prefix}.blocks"
  block_idx = 0

  for outer_layer in tf_block:

      layernorm_idx = 1
      mlp_layer_idx = 1

      if isinstance(outer_layer, SwinTransformerBlock3D):
          for inner_layer in outer_layer.layers:
        
              # Layer norm.
              if isinstance(inner_layer, tf.keras.layers.LayerNormalization):
                  #print("layer norm")
                  layer_norm_prefix = (
                      f"{common_prefix}.{block_idx}.norm{layernorm_idx}"
                  )
                  inner_layer.gamma.assign(
                      tf.Variable(
                          np_state_dict[f"{layer_norm_prefix}.weight"]
                      )
                  )
                #   print(i)
                #   i += 1

                  model_layers[f"{layer_norm_prefix}.weight"] = inner_layer.gamma.name 


                  inner_layer.beta.assign(
                      tf.Variable(np_state_dict[f"{layer_norm_prefix}.bias"])
                  )
                #   print(i)
                #   i += 1  
                  model_layers[f"{layer_norm_prefix}.bias"] = inner_layer.beta.name 
                  layernorm_idx += 1

              # Window attention.
              elif isinstance(inner_layer, WindowAttention3D):
                  #print("window attention")
                  attn_prefix = f"{common_prefix}.{block_idx}.attn"

                  # Relative position.
                  inner_layer.relative_position_bias_table = (
                      modify_tf_block(
                          inner_layer.relative_position_bias_table,
                          np_state_dict[
                              f"{attn_prefix}.relative_position_bias_table"
                          ] ,
                          wn =f"{attn_prefix}.relative_position_bias_table" ,
                      )
                  )
                  inner_layer.relative_position_index = (
                      modify_tf_block(
                          inner_layer.relative_position_index,
                          np_state_dict[
                              f"{attn_prefix}.relative_position_index"
                          ],
                          wn = f"{attn_prefix}.relative_position_index"
                      )
                  )

                  # QKV.
                  inner_layer.qkv = modify_tf_block(
                      inner_layer.qkv,
                      np_state_dict[f"{attn_prefix}.qkv.weight"],
                      np_state_dict[f"{attn_prefix}.qkv.bias"],
                      wn = f"{attn_prefix}.qkv.weight",
                       bn = f"{attn_prefix}.qkv.bias"
                  )

                  # Projection.
                  inner_layer.proj = modify_tf_block(
                      inner_layer.proj,
                      np_state_dict[f"{attn_prefix}.proj.weight"],
                      np_state_dict[f"{attn_prefix}.proj.bias"],
                      wn = f"{attn_prefix}.proj.weight",
                      bn = f"{attn_prefix}.proj.bias"
                  )

              # MLP.
              elif isinstance(inner_layer, tf.keras.Model):
                  #print("mlp")
                  mlp_prefix = f"{common_prefix}.{block_idx}.mlp"
                  for mlp_layer in inner_layer.layers:
                      if isinstance(mlp_layer, tf.keras.layers.Dense):
                          mlp_layer = modify_tf_block(
                              mlp_layer,
                              np_state_dict[
                                  f"{mlp_prefix}.fc{mlp_layer_idx}.weight"
                              ],
                              np_state_dict[
                                  f"{mlp_prefix}.fc{mlp_layer_idx}.bias"
                              ],
                              wn =  f"{mlp_prefix}.fc{mlp_layer_idx}.weight" ,
                              bn =  f"{mlp_prefix}.fc{mlp_layer_idx}.bias"
                          )
                          mlp_layer_idx += 1

          block_idx += 1
  return tf_block


def main(args):
    #print("Converting ",args.model_name)

    cfg_method = model_configs.MODEL_MAP[args.model_name]
    cfg = cfg_method()

    name = cfg["name"]
    link = cfg['link']
    del cfg["name"]
    del cfg['link']

    #print("Instantiating PyTorch model...")
    sys.path.append("..")
    pt_model = SwinTransformer3D_pt(**cfg)

    # command = f"wget {link} -O {name}.pth"
    # os.system(command)
    # #print("downloaded")

    if not os.path.exists(f"{name}.pth"):
        download_weight_command = f"wget {link} -O {name}.pth"
        os.system(download_weight_command)
        #print("downloaded")
    else:
        print("file exists")
        
    

    checkpoint = torch.load(f'{name}.pth')

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'backbone' in k:
            name = k[9:]
            new_state_dict[name] = v 

    pt_model.load_state_dict(new_state_dict) 

    # dummy_x = torch.rand(1, 3, 8, 224, 224)
    # logits = pt_model(dummy_x)
    # #print(logits.shape)

    input = tf.random.normal((1,3, 8, 224,224), dtype='float64')
    tf_model = SwinTransformer3D(**cfg)

    tf_model.compile()  # delete

    _ =  tf_model(input)

    # #print(_.shape)

    # Load the PT params.
    np_state_dict = pt_model.state_dict()
    np_state_dict = {k: np_state_dict[k].numpy() for k in np_state_dict}

    #print("Beginning parameter porting process...")

    #projection
    tf_model.projection.layers[0] = modify_tf_block(tf_model.projection.layers[0]
            ,
            np_state_dict["patch_embed.proj.weight"],
            np_state_dict["patch_embed.proj.bias"],
            bn = "patch_embed.proj.bias",
            wn  = "patch_embed.proj.weight")
    
    tf_model.projection.layers[1] = modify_tf_block(
        tf_model.projection.layers[1],
        np_state_dict["patch_embed.norm.weight"],
        np_state_dict["patch_embed.norm.bias"],
        wn = "patch_embed.norm.weight",
        bn = "patch_embed.norm.bias")
    
    # layer_normalization

    layer_normalization_idx = -1

    tf_model.layers[layer_normalization_idx] = modify_tf_block(
        tf_model.layers[layer_normalization_idx] ,
        np_state_dict["norm.weight"],
        np_state_dict["norm.bias"],
        wn = "norm.weight",
        bn = "norm.bias")
    
    # swin layers
    for i in range(2, len(tf_model.layers) - 1):
        #print(tf_model.layers[i])
        _ = modify_swin_blocks(np_state_dict,
                            f"layers.{i-2}",
                            tf_model.layers[i].layers)
        #print()
    
    #print("Porting successful, serializing TensorFlow model...")
    save_path = os.path.join(os.getcwd(), f"{args.model_name}_tf")

    _ =  tf_model(input)

    tf_model.save(save_path)
    # tf_model.save_weights('./checkpoints/my_checkpoint')




if __name__ == "__main__":
    args = parse_args()
    main(args)

    with open('data.json', 'w') as fp:
        json.dump(model_layers, fp)
    