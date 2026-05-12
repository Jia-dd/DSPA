# Diverse Style Prompt Assistant for Generalized Remote Sensing Object Classification
This repository includes introductions and implementation of **Diverse Style Prompt Assistant for Generalized Remote Sensing Object Classification** in PyTorch.

# Datasets

We conduct experiments using three remote sensing datasets: HRRSD, DOTA, and DIOR.

Remote sensing objects are cut out from object detection ground truth.

We select the common categories of these three datasets as experimental data.

```python
classnames = [
    'baseball diamond',
    'basketball court',
    'bridge',
    'ground track field',
    'harbor',
    'plane',
    'ship',
    'small vehicle',
    'storage tank',
    'tennis court'
]
```

You can use your own dataset and replace the classnames in `train.py` with your own list of class names.

# Preparation

You need to download the pre trained models of CLIP (ViT-B/32) and ResNet50 first.

Then create two new folders under the project folder, `pretrained_clip` and `pretrained-resnet50`, and place the downloaded model files `ViT-B-32.pt` and `resnet50.pt` in the two folders respectively.

# Train and Inference

### Train

Run `run_train.sh` directly.

You can modify the training parameters in `configs/config.yaml` and `run_train.sh`.

### Inference

Run `run_predict.sh` directly.

You can modify the inference parameters in `run_predict.sh`.
