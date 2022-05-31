# HARL - Heuristic Attention Representation Learning for Self-Supervised of Visual Representations

<span style="color: red"><strong> </strong></span> We have released a Pytorch Lightning implementation; (along with checkpoints)</a>.

<span style="color: red"><strong> </strong></span> Colabs for <a href=""> HARL framework </a> are added, see <a href="">here</a>.

<div align="center">
  <img width="50%" alt="HARL Framework Illustration" src="images/HARL_framework.png">
</div>
<div align="center">
  An illustration of HARL Framework (from <a href="https://www.hh-ri.com/2022/05/30/heuristic-attention-representation-learning-for-self-supervised-pretraining/">our blog here</a>).
</div>

## HARL Pre-trained models  

<a href="colabs/finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

We opensourced total 4 pretrained models here, corresponding to those in Table 1 of the <a href="">HARL</a> paper:

|   Depth | Width   | SK    |   Param (M)  |    Linear eval |   Supervised |
|--------:|--------:|------:|--------:|-------------:|--------------:|
|      50 | 1X      | False |    24 |        74.0 |          76.6 |


These checkpoints are stored in Google Drive Storage:


We also provide examples on how to use the checkpoints in `colabs/` folder.

## HARL Pre-trained models

The pre-trained models (base network with linear classifier layer) can be found below.

|                             Model checkpoint and hub-module                             |     ImageNet Top-1     |
|-----------------------------------------------------------------------------------------|------------------------|
|[ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) |          74.0          |




## Enviroment setup


Our code can also run on a *single* GPU or *multi-GPUs* GPUs.

The code is compatible with different Pytorch lightning versions . See requirements.txt for all prerequisites, and you can also install them using the following command.

```
pip install -r requirements.txt
```

## Pretraining

The following command can be used to pretrain a ResNet-50 on ImageNet (which reflects the default hyperparameters in our paper):

```
bash MNCRL.py --train_mode=pretrain \
  --train_batch_size=4096 --train_epochs=1000 
  --learning_rate=0.075 --learning_rate_scaling=sqrt --weight_decay=1e-4 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --train_summary_steps=0
```

A batch size of 4096 requires 8 A100 GPUs with 80G of VRAM. 1000 epochs takes around 149 hour. Note that learning rate of 0.2 with `learning_rate_scaling=linear` is equivalent to that of 0.075 with `learning_rate_scaling=sqrt` when the batch size is 4096. However, using sqrt scaling allows it to train better when smaller batch size is used.

## Finetuning the linear head (linear eval)

To fine-tune a linear head (with a single GPU), try the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \

```
As a reference, the above runs on CIFAR-10 should give you around 91% accuracy, though it can be further optimized.



For fine-tuning a linear head on ImageNet using GPUs, first set the `CHKPT_DIR` to pretrained model dir and set a new `MODEL_DIR`, then use the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=1e-6 \
  --train_epochs=90 --train_batch_size=4096 --warmup_epochs=0 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
```

As a reference, the above runs on ImageNet should give you around 64.5% accuracy.

## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset): simply set `dataset=imagenet2012_subset/1pct` and `dataset=imagenet2012_subset/10pct` in the command line for fine-tuning on these subsets.

You can also find image IDs of these subsets in `imagenet_subsets/`.

To fine-tune the whole network on ImageNet (1% of labels), refer to the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=-1 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head_supervised)' \
  --global_bn=True --optimizer=lars --learning_rate=0.005 \
  --learning_rate_scaling=sqrt --weight_decay=0 \
  --train_epochs=60 --train_batch_size=1024 --warmup_epochs=0 \
  --dataset=imagenet2012_subset/1pct --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0 \
  --num_proj_layers=3 --ft_proj_selector=1
```

Set the `checkpoint` to those that are only pre-trained but not fine-tuned. Given that SimCLRv1 checkpoints do not contain projection head, it is recommended to run with SimCLRv2 checkpoints (you can still run with SimCLRv1 checkpoints, but `variable_schema` needs to exclude `head`). The `num_proj_layers` and `ft_proj_selector` need to be adjusted accordingly following SimCLRv2 paper to obtain best performances.

## Other resources

### Our *offical* implementations in Different Frameworks

(Feel free to share your implementation by creating an issue)

Implementations in Pytorch Lightning:
* [Our Official Implementation]([https://github.com/TranNhiem/Heuristic_Attention_Represenation_Learning_SSL_Pytorch])


Implementations in Tensorflow 2:
* [Our Official Implementation]([https://github.com/TranNhiem/Heuristic_At- 432 tention_Representation_Learning_SSL_Tensorflow])


## Known issues


* **Pretrained models / Checkpoints**: HARL are pretrained with different weight decays, so the pretrained models from the two versions have very different weight norm scales. For fine-tuning the pretrained models from both versions, it is fine if you use an LARS optimizer, but it requires very different hyperparameters (e.g. learning rate, weight decay) if you use the momentum optimizer. So for the latter case, you may want to either search for very different hparams according to which version used, or re-scale th weight (i.e. conv `kernel` parameters of `base_model` in the checkpoints) to make sure they're roughly in the same scale.

## Cite

[HARL paper]():

```
@article{
}
```




