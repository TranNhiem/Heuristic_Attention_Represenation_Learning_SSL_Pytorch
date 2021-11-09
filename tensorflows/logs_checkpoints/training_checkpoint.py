
import tensorflow as tf
import os
# Configure Saving Model --> Chief worker will reponsible for saving model
from multiprocessing import util

class checkpoint():
    def __init__(self, strategy, checkpoint_path):
        self.strategy = strategy
        self.checkpoint_path=checkpoint_path
        self.checkpoint_dir_encoder = os.path.join(checkpoint_path, 'ckpt')
        self.checkpoint_dir_mlp = os.path.join(checkpoint_path, 'MLP_ckpt')

    @classmethod()
    def _is_chief(self, task_type, task_id):
        return task_type is None or task_type == 'chief' or (task_type == 'worker' and
                                                             task_id == 0)

    @classmethod()
    def _get_temp_dir(self, dirpath, task_id):
        base_dirpath = 'workertemp_' + str(task_id)
        temp_dir = os.path.join(dirpath, base_dirpath)
        tf.io.gfile.makedirs(temp_dir)
        return temp_dir

    @classmethod()
    def write_filepath(self, filepath, task_type, task_id):
        dirpath = os.path.dirname(filepath)
        base = os.path.basename(filepath)
        if not self._is_chief(task_type, task_id):
            dirpath = self._get_temp_dir(dirpath, task_id)
        return os.path.join(dirpath, base)

    def checkpoint(self, encoder, MLP):

        epoch = tf.Variable(initial_value=tf.constant(
            0, dtype=tf.dtypes.int64), name='epoch')
        step_in_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64),
                                    name='step_in_epoch')
        task_type, task_id = (
            self.strategy.cluster_resolver.task_type, self.strategy.cluster_resolver.task_id)

        checkpoint_encoder = tf.train.Checkpoint(
            model=encoder, epoch=epoch, step_in_epoch=step_in_epoch)

        write_checkpoint_dir = self.write_filepath(
            self.checkpoint_dir_encoder, task_type, task_id)

        write_checkpoint_dir_1 = self.write_filepath(
            self.checkpoint_dir_mlp, task_type, task_id)

        checkpoint_manager_encoder = tf.train.CheckpointManager(
            checkpoint_encoder, directory=write_checkpoint_dir, max_to_keep=1)

        checkpoint_MLP = tf.train.Checkpoint(
            model=MLP, epoch=epoch, step_in_epoch=step_in_epoch)

        checkpoint_manager_MLP = tf.train.CheckpointManager(
            checkpoint_MLP, directory=write_checkpoint_dir_1, max_to_keep=1)

        return checkpoint_manager_encoder, checkpoint_manager_MLP, write_checkpoint_dir, write_checkpoint_dir_1
