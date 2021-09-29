import wandb
from utils.args import parse_args
from Data_utils.datasets import SEED
from Data_utils.datasets import CIFAR100_dataset
from tensorflow.keras import optimizers
from tensorflow.python.keras.backend import dropout, learning_phase
import tensorflow_addons as tfa
from Neural_Net_Architecture.Convnet_Transformer.perceiver_compact_Conv_transformer_VIT_architecture import conv_transform_VIT

import argparse
from tensorflow.keras.optimizers import schedules
from Training_strategy.learning_rate_optimizer_weight_decay_schedule import WarmUpAndCosineDecay, get_optimizer
from wandb.keras import WandbCallback
import tensorflow as tf

#import tensorflow as tf

wandb.login()

# Setting GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[0:8], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()
Auto = tf.data.experimental.AUTOTUNE


# Setting GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:

    try:
        tf.config.experimental.set_visible_devices(gpus[0:8], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()

# Try to keep latten array small
input_shape = (32, 32, 3)
IMG_SIZE = 32
num_class = 100
# Patches unroll for ViT and Normal transformer
# patch_size = 4
# num_patches = (IMG_SIZE//patch_size)**2
# data_dim = num_patches

num_conv_layers = 2  # for unroll patches -- Overlap
spatial2projection_dim = [128, 256]  # This equivalent to # filters
position_embedding_option = True
latten_dim = 128  # size of latten array --> (N)
projection_dim = 256
dropout = 0.2
stochastic_depth_rate = 0.1
# Learnable array
# (NxD) #--> OUTPUT( [Q, K][Conetent information, positional])
# latten_array = latten_dim * projection_dim

num_multi_heads = 8  # --> multhi Attention Module to processing inputs
# Encoder -- Decoder are # --> Increasing block create deeper Transformer model
NUM_TRANSFORMER_BLOCK = 4
# Corresponding with Depth of self-attention
# Model depth stack multiple CrossAttention +self-trasnformer_Block
NUM_MODEL_LAYERS = 4

# 2 layer MLP Dense with number of Unit= pro_dim
FFN_layers_units = [projection_dim, projection_dim]
classification_head = [projection_dim, num_class]

print(f"Image size: {IMG_SIZE} X {IMG_SIZE} = {IMG_SIZE ** 2}")
# print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
# print(f"Patches per image: {num_patches}")
# print(
#     f"Elements per patch [patch_size*patch_size] (3 channels RGB): {(patch_size ** 2) * 3}")

# print(f"Data array shape: {num_patches} X {projection_dim}")


with strategy.scope():

    def main(args):

        BATCH_SIZE = args.train_batch_size
        EPOCHS = args.train_epochs

        # Prepare data training
        data = CIFAR100_dataset(BATCH_SIZE, IMG_SIZE)
        num_images = data.num_train_images
        train_ds, test_ds = data.supervised_train_ds_test_ds()

        # Create model Architecutre
        # Noted of Input pooling mode 2D not support in current desing ["1D","sequence_pooling" ]

        conv_VIT_model = conv_transform_VIT(num_class,IMG_SIZE, num_conv_layers,   spatial2projection_dim,position_embedding_option,
                                            NUM_TRANSFORMER_BLOCK, num_multi_heads, projection_dim,
                                            FFN_layers_units, dropout,
                                            classification_head, include_top=True, pooling_mode="sequence_pooling",
                                            stochastic_depth=False, stochastic_depth_rate=stochastic_depth_rate)

        conv_VIT_model(tf.keras.Input((input_shape)))
        conv_VIT_model.summary()

        # Initialize the Random weight
        x = tf.random.normal((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
        h = conv_VIT_model(x, training=False)
        print("Succeed Initialize online encoder")
        print(f"Conv_Perciever encoder OUTPUT: {h.shape}")

        num_params_f = tf.reduce_sum(
            [tf.reduce_prod(var.shape) for var in conv_VIT_model.trainable_variables])
        print('The encoders have {} trainable parameters each.'.format(num_params_f))

        '''
        # Configure Logs recording during training
        
        #Training Configure

        configs = {
            "Model_Arch": "Conv_Perceiver_arch",
            "DataAugmentation_types": "None for testing",
            "Dataset": "Cifar100",
            "IMG_SIZE": IMG_SIZE,
            "Epochs": EPOCHS,
            "Batch_size": BATCH_SIZE,
            "Learning_rate": "1e-3*Batch_size/512",
            "Optimizer": "AdamW",
            "SEED": SEED,
            "Loss type": "Cross_entropy_loss",
        }

        wandb.init(project="heuristic_attention_representation_learning",
                   sync_tensorboard=True, config=configs)

        # Model Hyperparameter Defined Primary
        # 1. Define init
        # base_lr = 1e-3
        # weight_decay = 1e-6
        # # 2. Schedule init
        # step = tf.Variable(0, trainable=False)
        # schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        #     [10000, 15000], [1e-0, 1e-1, 1e-2])
        # lr_schedule = 1e-3*schedule(step)
        # def weight_decay_sche(): return 1e-4 * schedule(step)

        # optimizer = tfa.optimizers.LAMB(
        #     learning_rate=init_lr, weight_decay_rate=weight_decay_sche)

        # optimizer = tfa.optimizers.SGDW(
        #     learning_rate=lr_rate, momentum=0.9, weight_decay=weight_decay)

        # optimizer = tfa.optimizers.AdamW(
        #     learning_rate=init_lr, weight_decay=weight_decay)

        # Custom Define Hyperparameter

        # 3. Schedule CosineDecay warmup
        base_lr = 0.3
        lr_rate = WarmUpAndCosineDecay(base_lr, num_images)
        optimizers = get_optimizer(lr_rate)
        AdamW = optimizers.optimizer_weight_decay

        # model compile
        conv_perceiver_model.compile(optimizer=AdamW,
                                     loss=tf.keras.losses.CategoricalCrossentropy(),
                                     metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                                              tf.keras.metrics.TopKCategoricalAccuracy(5, name="top5_acc")])

        # MODEL TRAINING

        conv_perceiver_model.fit(train_ds, epochs=EPOCHS,
                                 validation_data=test_ds, callbacks=[WandbCallback()])  # callbacks=callbacks_list,
        '''

    if __name__ == '__main__':

        args = parse_args()

        main(args)
