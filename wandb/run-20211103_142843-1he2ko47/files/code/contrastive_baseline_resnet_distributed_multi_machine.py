import os
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras import mixed_precision
import argparse
from imutils import paths
from tensorflows.losses_optimizers.learning_rate_optimizer_weight_decay_schedule import WarmUpAndCosineDecay, get_optimizer
from tensorflows.Data_utils.byol_simclr_imagenet_data import imagenet_dataset
from tensorflows.losses_optimizers.self_supervised_losses import nt_xent_asymetrize_loss_v2
################################################
# Configuration
################################################
parser = argparse.ArgumentParser()
# Configure for training
parser.add_argument('--train_epochs', type=int, default=600,
                    help='Number of iteration')
parser.add_argument('--Batch_size', default=256, type=int,)
parser.add_argument('--IMG_SIZE', default=224, type=int,)
parser.add_argument('--seed', default=26, type=int,)
parser.add_argument('--project_dim', default=256, type=int)
# simclr 0.05 best contrastvie acc [0.05, 0,5, 0,1]
parser.add_argument('--temperature', default=0.1, type=float)

# Configure Learning Rate and Optimizer
# In optimizer we will have three Option ('Original Configure', 'Weight Decay', 'Gradient Centralization')

parser.add_argument('--learning_rate_scaling', metavar='learning_rate', default='linear',
                    choices=['linear', 'sqrt', 'no_scale', ])

parser.add_argument('--optimizer', type=str, default="LARS", help="Optimization for update the Gradient",
                    choices=['Adam', 'SGD', 'LARS', 'AdamW', 'SGDW', 'LARSW',
                             'AdamGC', 'SGDGC', 'LARSGC', 'AdamW_GC', 'SGDW_GC', 'LARSW_GC'])
parser.add_argument('--momentum', type=float, default=0.9,
                    help="Momentum manage how fast of update Gradient")

parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help="weight_decay to penalize the update gradient")
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Warmup the learning base period -- this Larger --> Warmup more slower')


# Configure for Distributed training
parser.add_argument('--mode', type=str, default="mix_pre_fp16_v1", choices=["mix_precision_fp16_", "mix_precision_fp16", "mix_pre_fp16_v1", "mix_pre_fp16_v1_", "mix_per_pack_NCCL"],
                    help='mix_precision_implementation or orignal mode')
parser.add_argument('--communication_method', type=str,
                    default="NCCL", choices=["NCCL", "auto", ])

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

    resnet_base = tf.keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=(args.IMG_SIZE, args.IMG_SIZE, 3))
    # Enable to train the whole
    resnet_base.trainable = True
    last_layer = resnet_base.layers[-1].output
    x = tf.keras.layers.GlobalAveragePooling2D()(last_layer)
    model = tf.keras.Model(inputs=resnet_base.input,
                           outputs=x, name="Resnet50_keras_model")

    return model

# 512 (h) -> 256 -> 128 (z)

def MLP_model(projection_dim): 
    inputs= tf.keras.layers.Input(( 2048))
    fc1_= tf.keras.layers.Dense(units=projection_dim,)(inputs)
    bn= tf.keras.layers.BatchNormalization()(fc1_)
    relu=tf.nn.relu(bn)
    fc1_= tf.keras.layers.Dense(units=projection_dim,)(relu)
    relu=tf.nn.relu(fc1_)
    out=tf.keras.layers.Dense(units=projection_dim/2,)(relu)
    model= tf.keras.Model(inputs= inputs, outputs= out, name="MLP_model")
    return model
# class MLP_model(tf.keras.Model):
#     def __init__(self, Projection_dim):
#         self.pro_dim = Projection_dim
#         super(MLP_model, self).__init__()
#         self.fc1 = tf.keras.layers.Dense(units=self.pro_dim)
#         self.bn = tf.keras.layers.BatchNormalization()
#         self.fc2 = tf.keras.layers.Dense(units=self.pro_dim/2)

#     def call(self, inp, training=False):
#         x = self.fc1(inp)
#         x = self.bn(x, training=training)
#         x = tf.nn.relu(x)
#         x = self.fc1(x)
#         x = tf.nn.relu(x)
#         x = self.fc2(x)
#         return x


################################################
# Multi-Workers Distributed Training Loop
################################################

strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options)

with strategy.scope():

    def main(args):

        per_worker_batch_size = args.Batch_size
        num_workers = 2  # len(tf_config['cluster']['worker'])
        global_batch_size = per_worker_batch_size * num_workers
        temperature = args.temperature
        # Configure Training Data -- Distribute training data over multi-machine
        train_image_path = "/data/rick109582607/Desktop/TinyML/self_supervised/imagenet_1k/train/"
        val_image_path = "/data/rick109582607/Desktop/TinyML/self_supervised/imagenet_1k/val/"
        total_training_sample = len(list(paths.list_images(train_image_path)))

        dataset = imagenet_dataset(
            args.IMG_SIZE, global_batch_size, args.seed, train_image_path, val_image_path)
        multi_worker_dataset = strategy.distribute_datasets_from_function(
            lambda input_context: dataset.simclr_inception_style_crop(input_context))

        # Configure Neural Net architecture
        resnet_encoder = keras_Resnet_encoder(args)
        print("Resnet Encoder architecture")
        resnet_encoder.summary()
        MLP = MLP_model(args.project_dim)
        #MLP.build(input_shape=(None, 2048))
        print("MLP Model architecture")
        MLP.summary()

        # Get model loss and Optimizer
        # Deeper Investiage distribute batch
        def co_distributed_loss(p, z, temperature, distribute_batch):
            per_batch_co_distribute_loss = nt_xent_asymetrize_loss_v2(
                p, z, temperature, distribute_batch)
            return tf.nn.compute_average_loss(per_batch_co_distribute_loss, global_batch_size=distribute_batch)
        #train_loss = tf.keras.metrics.Mean(name="train_loss")
        # Cosine decay warmup learning
        base_lr = 0.3
        scale_lr = args.learning_rate_scaling
        warmup_epochs = args.warmup_epochs
        train_epochs = args.train_epochs
        lr_schedule = WarmUpAndCosineDecay(
            base_lr, global_batch_size, total_training_sample, scale_lr, warmup_epochs, train_epochs)
        Optimizer_type = args.optimizer
        optimizers = get_optimizer(lr_schedule, Optimizer_type)
        LARS = optimizers.original_optimizer(args)
        LARS_mix_percision = mixed_precision.LossScaleOptimizer(LARS)
        # Tracking Experiment
        configs = {
            "Model_Arch": "ResNet50",
            "Training mode": "SSL",
            "DataAugmentation_types": "SimCLR",
            "Dataset": "ImageNet1k",
            "IMG_SIZE": args.IMG_SIZE,
            "Epochs": args.train_epochs,
            "Batch_size": global_batch_size,
            "Learning_rate": args.learning_rate_scaling,
            "Optimizer": args.optimizer,
            "SEED": args.seed,
            "Loss type": "NCE_Loss Temperature",
        }

        wandb.init(project="heuristic_attention_representation_learning",
                   sync_tensorboard=True, config=configs)

        # Configure training steps

        @tf.function()
        def train_step(ds_one, ds_two):  # (bs, 32, 32, 3), (bs)
            # Forward pass
            with tf.GradientTape() as tape:
                # (bs, 512)
                rep_ds1 = resnet_encoder(ds_one, training=True)  # (bs,)
                rep_ds1_projt = MLP(rep_ds1)
                rep_ds2 = resnet_encoder(ds_two, training=True)  # (bs, )
                rep_ds2_projt = MLP(rep_ds2)
                loss = co_distributed_loss(
                    rep_ds1_projt, rep_ds2_projt, temperature, global_batch_size)

            # Backward pass Mixpercision Gradient

            # Backbone Encoder
            fp32_grads = tape.gradient(
                loss, resnet_encoder.trainable_variables)

            fp16_grads = [tf.cast(grad, 'float16') for grad in fp32_grads]

            hints = tf.distribute.experimental.CollectiveHints(
                bytes_per_pack=32 * 1024 * 1024)

            all_reduce_fp16_grads = tf.distribute.get_replica_context().all_reduce(
                tf.distribute.ReduceOp.SUM, fp16_grads, options=hints)

            all_reduce_fp32_grads = [
                tf.cast(grad, 'float32') for grad in all_reduce_fp16_grads]

            all_reduce_fp32_grads = LARS_mix_percision.get_unscaled_gradients(
                all_reduce_fp32_grads)

            LARS_mix_percision.apply_gradients(zip(
                all_reduce_fp32_grads, resnet_encoder.trainable_variables), experimental_aggregate_gradients=False)

            # MLP Projection
            fp32_grads = tape.gradient(loss, MLP.trainable_variables)

            fp16_grads = [tf.cast(grad, 'float16') for grad in fp32_grads]

            hints = tf.distribute.experimental.CollectiveHints(
                bytes_per_pack=32 * 1024 * 1024)

            all_reduce_fp16_grads = tf.distribute.get_replica_context().all_reduce(
                tf.distribute.ReduceOp.SUM, fp16_grads, options=hints)

            all_reduce_fp32_grads = [
                tf.cast(grad, 'float32') for grad in all_reduce_fp16_grads]

            all_reduce_fp32_grads = LARS_mix_percision.get_unscaled_gradients(
                all_reduce_fp32_grads)

            LARS_mix_percision.apply_gradients(zip(
                all_reduce_fp32_grads, MLP.trainable_variables), experimental_aggregate_gradients=False)

            return loss

        @tf.function
        def distributed_train_step(ds_one, ds_two):
            per_replica_losses = strategy.run(
                train_step, args=(ds_one, ds_two))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

        # Configure Saving Model --> Chief worker will reponsible for saving model
        from multiprocessing import util

        checkpoint_path = './model_checkpoints/'
        checkpoint_dir_encoder = os.path.join(checkpoint_path, 'ckpt')
        checkpoint_dir_mlp = os.path.join(checkpoint_path, 'MLP_ckpt')

        # Helper Function to define chief worker and saving file

        def _is_chief(task_type, task_id):
            return task_type is None or task_type == 'chief' or (task_type == 'worker' and
                                                                 task_id == 0)

        def _get_temp_dir(dirpath, task_id):
            base_dirpath = 'workertemp_' + str(task_id)
            temp_dir = os.path.join(dirpath, base_dirpath)
            tf.io.gfile.makedirs(temp_dir)
            return temp_dir

        def write_filepath(filepath, task_type, task_id):
            dirpath = os.path.dirname(filepath)
            base = os.path.basename(filepath)
            if not _is_chief(task_type, task_id):
                dirpath = _get_temp_dir(dirpath, task_id)
            return os.path.join(dirpath, base)

        # Saving and restore Checkpoint
        epoch = tf.Variable(initial_value=tf.constant(
            0, dtype=tf.dtypes.int64), name='epoch')
        step_in_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64),
                                    name='step_in_epoch')
        task_type, task_id = (strategy.cluster_resolver.task_type,
                              strategy.cluster_resolver.task_id)

        checkpoint_encoder = tf.train.Checkpoint(
            model=resnet_encoder, epoch=epoch, step_in_epoch=step_in_epoch)

        write_checkpoint_dir = write_filepath(
            checkpoint_dir_encoder, task_type, task_id)

        write_checkpoint_dir_1 = write_filepath(
            checkpoint_dir_mlp, task_type, task_id)

        checkpoint_manager_encoder = tf.train.CheckpointManager(
            checkpoint_encoder, directory=write_checkpoint_dir, max_to_keep=1)

        checkpoint_MLP = tf.train.Checkpoint(
            model=MLP, epoch=epoch, step_in_epoch=step_in_epoch)

        checkpoint_manager_MLP = tf.train.CheckpointManager(
            checkpoint_MLP, directory=write_checkpoint_dir_1, max_to_keep=1)

        for epoch in range(args.train_epochs):
            total_loss = 0.0
            num_batches = 0
            for _, (ds_one, ds_two) in enumerate(multi_worker_dataset):

                total_loss += distributed_train_step(
                    ds_one, ds_two)

                num_batches += 1

            train_loss = total_loss/num_batches

            if epoch % 2 == 0:

                wandb.log({
                    "epochs": epoch,
                    "train_loss": train_loss,
                })

                template = ("Epoch {}, Train Loss: {},  ")
                print(template.format(epoch+1, train_loss,))

                # Saving the Model
                checkpoint_manager_encoder.save()
                checkpoint_manager_MLP.save()
                # keras saving .h5 format
                save_path = "./model_checkpoints/h5_format/"
                # try:
                #     os.mkdir(save_path)
                # except:
                #     pass
                save = 'encoder_resnet50_' + str(epoch)
                save_1 = "mlp_" + str(epoch)
                path_encoder = os.path.join(save_path, save)
                path_mlp = os.path.join(save_path, save_1)

                resnet_encoder.save_weights(path_encoder)
                MLP.save_weights(path_mlp)

                if not _is_chief(task_type, task_id):
                    tf.io.gfile.rmtree(write_checkpoint_dir)
                    tf.io.gfile.rmtree(write_checkpoint_dir_1)

    if __name__ == '__main__':

        main(args)
