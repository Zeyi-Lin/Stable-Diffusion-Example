# Stbale Diffusion Examples

## 1. 环境安装

```bash
pip install -r requirements.txt
```

## （可选）模型与数据集准备

- 如果你与HuggingFace网络连接顺畅，可以直接运行训练代码；
- 如果不，[百度云](https://pan.baidu.com/s/1Yu5HjXnHxK0Wgymc8G-g5g?pwd=gtk8)，提取码: gtk8；将压缩文件下载并解压后，放到与代码同一目录下，并将`sd_config.py`中`pretrained_model_name_or_path`和`dataset_name`的default进行修改：

```python
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="./naruto-blip-captions",
    )
```


## 2. 训练

SD1.5 + 火影任务数据集：

```python
python train_sd1-5_naruto.py \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --seed=42 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model"
```

## 3. 推理

```bash
python predict.py
```
