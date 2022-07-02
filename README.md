# HARL - Heuristic Attention Representation Learning for Self-Supervised Pretraining

<span style="color: red"><strong> </strong></span> This is offical implemenation of HARL framework</a>.

<span style="color: red"><strong> </strong></span> Colabs for <a href=""> HARL framework </a> are added, see <a href="">here</a>.

<div align="center">
  <img width="50%" alt="HARL Framework Illustration" src="images/HARL_framework.png">
</div>
<div align="center">
  End-to-End HARL Framework (from <a href="https://www.hh-ri.com/2022/05/30/heuristic-attention-representation-learning-for-self-supervised-pretraining/">our blog here</a>).
</div>

# Table of Contents

  - [Installation](#installation)
  - [Dataset -- Heuristic Mask Retrieval Techniques](#Generating-Heuristic-binary-mask-Natural-image)
  - [Configure Self-Supervised Pretraining](#Setup-self-supervised-pretraining)
    - [Dataset](#Natural-Image-Dataset)
    - [Hyperamters Setting](#Important-Hyperparameter-Setting)
    - [Choosing # augmentation Strategies](#Number-Augmentation-Strategies)
    - [Single or Multi GPUs](#Single-Multi-GPUS)
  - [Pretrained model](#model-weights)
  - [Downstream Tasks](#running-tests)
     - [Image Classification Tasks](#Natural-Image-Classification)
     - [Other Vision Tasks](#Object-Detection-Segmentation)
  - [Contributing](#contributing)

## Installation

```
pip or conda installs these dependents in your local machine
```
### Requirements
* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm

## Dataset -- Heuristic Mask Retrieval Techniques

```
If you are using ImageNet 1k dataset for self-supervised pretraining. 
We provodie two sets of heuristic mask generated for whole ImageNet train set available for download. 

```
|                            Heuristic Mask Dataset                               |
|---------------------------------------------------------------------------------|
|[DRFI Mask](https://drive.google.com/file/d/1-WCt2a4jhLWhiyJbfwY7sPsXYQg2TjxF/view?usp=sharing)|
|[Unsupervised Deep Learning Mask](https://drive.google.com/file/d/1-Ph6f4lLVe9Og_6_Ko2vx4sSDYKO6b7C/view?usp=sharing)|

### Using Custome Dataset 
**1.Generating Heuristic Binary Mask Using Deep Learning method**

We created one python module that directly with the input directory of your dataset
then generate by providing the filename
'''
/heuristic_mask_techniques/Deeplearning_methods/DeepMask.py

'''

## Self-supervised Pretraining

###  Preparing  Dataset: 

**NOTE:** Currently, This repository support self-supervised pretraining on the ImageNet dataset. 
+ 1. Download ImageNet-1K dataset (https://www.image-net.org/download.php). Then unzip folder follow imageNet folder structure. 
+ 2. Please first Download the Mask dataset either DRFI or Deep Learning Masks.

###  in pretraining Flags: 
`
Naviaging to the 

bash_files/pretrain/imagenet/HARL.sh
`

**1 Changing the dataset directory according to your path**
    `
    --train_dir ILSVRC2012/train \
    --val_dir ILSVRC2012/val \
    --mask_dir train_binary_mask_by_USS \
    `
**2 Other Hyperparameters setting** 
  
  - Use a large init learning rate {0.3, 0.4} for `short training epochs`. This would archieve better performance, which could be hidden by the initialization if the learning rate is too small.Use a small init learning rate for Longer training epochs should use value around 0.2.

    `
    --max_epochs 100 \
    --batch_size 512 \
    --lr 0.5 \
    `
**3 Distributed training in 1 Note**

`
Controlling number of GPUs in your machine by change the --gpus flag
    --gpus 0,1,2,3,4,5,6,7\
    --accelerator gpu \
    --strategy ddp \
`
## HARL Pre-trained models  

We opensourced total 8 pretrained models here, corresponding to those in Table 1 of the <a href="">HARL</a> paper:

|   Depth | Width   | SK    |   Param (M)  | Pretrained epochs| SSL pretrained learning_rate |Projection head MLP Dimension| Heuristic Mask| Linear eval  |
|--------:|--------:|------:|--------:|-------------:|--------------:|--------:|-------------:|-------------:|
| [ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) | 1X | False | 24 | 1000 |  0.5| 256 |Deep Learning mask| 73.6 |     
| [ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) | 1X  | False | 24 | 1000 |  0.3 | 256 |Deep Learning mask|  73.8 |   
| [ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) | 1X  | False | 24 | 1000 |  0.2 | 512 |Deep Learning mask| 74.0 |   
| [ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) | 1X  | False | 24 | 300 |  0.3 | 256 |Deep Learning mask| 69.4 |     
| [ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) | 1X  | False | 24 | 300 |  0.4 | 512 |Deep Learning mask| 70.7 |     
| [ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) | 1X  | False | 24 | 300 |  0.5 | 512 |Deep Learning mask| 71.4 | 
| [ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) | 1X  | False | 24 | 100 |  0. | 512 |DRFI  mask| 61.2 |     
| [ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) | 1X  | False | 24 | 100 |  0.2 | 512 |Deep Learning mask| 62.0 | 

These checkpoints are stored in Google Drive Storage:

## Finetuning the linear head (linear eval)

To fine-tune a linear head (with a single GPU), try the following command:

For fine-tuning a linear head on ImageNet using GPUs, first set the `CHKPT_DIR` to pretrained model dir and set a new `MODEL_DIR`, then use the following command:
`
Stay tune! The instructions will update soon
`

## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset): simply set `dataset=imagenet2012_subset/1pct` and `dataset=imagenet2012_subset/10pct` in the command line for fine-tuning on these subsets.

You can also find image IDs of these subsets in `imagenet_subsets/`.

To fine-tune the whole network on ImageNet (1% of labels), refer to the following command:

`
Stay tune! The instructions will update soon
`

## Other resources

### Our *offical* implementations in Different Frameworks

(Feel free to share your implementation by creating an issue)

Implementations in Tensorflow 2:
* [Official Implementation](https://github.com/TranNhiem/Heuristic_Attention_Representation_Learning_SSL_Tensorflow)

## Known issues

* **Pretrained models / Checkpoints**: HARL are pretrained with different weight decays, so the pretrained models from the two versions have very different weight norm scales. For fine-tuning the pretrained models from both versions, it is fine if you use an LARS optimizer, but it requires very different hyperparameters (e.g. learning rate, weight decay) if you use the momentum optimizer. So for the latter case, you may want to either search for very different hparams according to which version used, or re-scale th weight (i.e. conv `kernel` parameters of `base_model` in the checkpoints) to make sure they're roughly in the same scale.

## Citation

`
@inproceedings{tran2022HARL,
  title={MASSRL},
  author={},
  booktitle={},
  pages={},
  year={2022}
}
`

