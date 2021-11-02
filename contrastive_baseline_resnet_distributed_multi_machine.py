import os 
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras import mixed_precision
import argparse
from losses_optimizers.learning_rate_optimizer_weight_decay_schedule import WarmUpAndCosineDecay, get_optimizer
from Data_utils.byol_simclr_imagenet_data import imagenet_dataset

################################################
# Configuration 
################################################
parser = argparse.ArgumentParser()
## Configure for training 
parser.add_argument('--train_epochs', type=int, default=600,
                        help='Number of iteration')
parser.add_argument('--Batch_size', default=250, type=int,)
parser.add_argument('--IMG_SIZE', default=224, type=int,)
parser.add_argument('--seed', default=26, type=int,)

## Configure Learning Rate and Optimizer 
# In optimizer we will have three Option ('Original Configure', 'Weight Decay', 'Gradient Centralization')

parser.add_argument('--learning_rate_scaling', metavar='learning_rate', default='linear',
                        choices=['linear', 'sqrt', 'no_scale', ])

parser.add_argument('--optimizer', type=str, default="LARSW_GC", help="Optimization for update the Gradient",
                    choices=['Adam', 'SGD', 'LARS', 'AdamW', 'SGDW', 'LARSW',
                                'AdamGC', 'SGDGC', 'LARSGC', 'AdamW_GC', 'SGDW_GC', 'LARSW_GC'])                              
parser.add_argument('--momentum', type=float, default=0.9,
                    help="Momentum manage how fast of update Gradient")

parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help="weight_decay to penalize the update gradient")
parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup the learning base period -- this Larger --> Warmup more slower')


## Configure for Distributed training
parser.add_argument('--mode', type=str, default="mix_pre_fp16_v1", choices=["mix_precision_fp16_", "mix_precision_fp16", "mix_pre_fp16_v1", "mix_pre_fp16_v1_", "mix_per_pack_NCCL"],
                    help='mix_precision_implementation or orignal mode')
parser.add_argument('--communication_method', type=str,
                    default="auto", choices=["NCCL", "auto", ])

args = parser.parse_args()

if args.communication_method == "NCCL":

    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

elif args.communication_method == "auto":
    communication_options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CollectiveCommunication.AUTO)



################################################
# Neural Net Encoder -- MLP
################################################

def keras_Resnet_encoder(args): 
    
    resnet_base=tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(args.IMG_SIZE, args.IMG_SIZE,3))
     # Enable to train the whole
    resnet_base.trainable = True
    last_layer= resnet_base.layers[-1].output
    x= tf.keras.layers.GlobalAveragePooling2D()(last_layer)
    model=tf.keras.Model(inputs=resnet_base.input, outputs=x, name="Resnet50_keras_model")
    
    return model

# 512 (h) -> 256 -> 128 (z)
class MLP(tf.keras.Model):
    def __init__(self, Projection_dim):
        self.pro_dim=Projection_dim
        super(MLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units= self.pro_dim)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(units= self.pro_dim/2)

    def call(self, inp, training=False):
        x = self.fc1(inp)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x


################################################
# Multi-Workers Distributed Training Loop 
################################################

strategy= tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

with strategy.scope(): 
    
    def main(args):

        per_worker_batch_size = args.Batch_size
        num_workers = 3  # len(tf_config['cluster']['worker'])
        global_batch_size = per_worker_batch_size * num_workers

        # Configure Training Data -- Distribute training data over multi-machine
        train_image_path= "train_data"
        val_image_path="test_data"
        
        dataset= imagenet_dataset(args.IMG_SIZE, global_batch_size,args.seed, train_image_path)
        multi_worker_dataset = strategy.distribute_datasets_from_function(
            lambda input_context: dataset.simclr_inception_style_crop(input_context))

