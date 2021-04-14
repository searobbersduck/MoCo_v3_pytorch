# MoCo_v3_pytorch

an unofficial implemenation of [moco v3](https://arxiv.org/pdf/2104.02057.pdf).

[An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.02057.pdf)

## reference link



## requirements

```
git submodule update --init --recursive
```

```
conda create --name mocov3 --file requirements.txt
source activate mocov3
```

## train

only use visual transformer as backbone, run as follows:

```
CUDA_VISIBLE_DEVICES=6 python main_mocov2_vit.py -a resnet18 --lr 0.03 --batch-size 1 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 --moco-k 65536  --epochs 10 /your_data_root
```

train moco v3:

```
CUDA_VISIBLE_DEVICES=6 python main_mocov3_vit.py -a resnet18 --lr 0.03 --batch-size 4096 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 --moco-k 65536  --epochs 10 /your_data_root
```

### use your datasets

change the dataset line(`train_dataset = ToyDS()`) in function `main_worker` in `main_mocov3.py`

### change visual transformer parameters

```
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
```

