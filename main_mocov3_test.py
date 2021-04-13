import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from datasets.toy_ds import ToyDS
import mocov3
import mocov3.builder
import mocov3 as moco
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'external_lib/vit-pytorch')))
from vit_pytorch import ViT

import torch
import torch.nn as nn

print("=> creating model '{}'".format('vit'))

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)


model = mocov3.builder.MoCo(
    v, 
    128, 32, 0.999, 0.07, False)
    # args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
print(model)


print('hello world!')