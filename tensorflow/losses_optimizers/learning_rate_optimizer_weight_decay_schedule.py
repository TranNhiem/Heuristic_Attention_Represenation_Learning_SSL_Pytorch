'''
Strategy Training Need to Implementation Effective
# In Contrastive SSL framework
************************************************************************************
Training Configure
************************************************************************************
1. Learning Rate
    + particular implementation : Scale Learning Rate Linearly with Batch_SIZE (With Cosine Learning Rate)
    + Warmup: Learning Implementation
    + Schedule Learning with Constrain-Update during training

2. Optimizer -- With & Without Gradient Centralize
    1.LARS_optimizer for Contrastive + Large batch_size
    2. SGD - RMSProp - Adam (Gradient Centralize)
    3. SGD -- RMSProp -- Adam (Weight Decay) (TFA)

3. Regularization Weight Decay
    weight decay: Start with 1e6

************************************************************************************
FineTuning Configure
************************************************************************************
1. Learning Rate

2. Optimizer (Regularization weight Decay)

'''

# 1 implementation Cosine Decay Learning rate schedule (Warmup Period)-- Implement
'''
[1] SimCLR [2] BYOL [Consisten Distillation Training]
in SimCLR & BYOL with warmup steps =10
'''

# Reference https://github.com/google-research/simclr/blob/dec99a81a4ceccb0a5a893afecbc2ee18f1d76c3/tf2/model.py


#import tensorflow.compat.v2 as tf
#import tensorflow_addons as tfa
#import tensorflow as tf


import tensorflow.keras.backend as K
from utils.args import parse_args
import numpy as np
import math
import tensorflow as tf
import tensorflow_addons as tfa
from losses.lars_optimizer_v2 import LARS_optimzer
args = parse_args()

# this helper function determine steps per epoch


def get_centralized_gradients(optimizer, loss, params):
    """Compute the centralized gradients.

    This function is ideally not meant to be used directly unless you are building a custom optimizer, in which case you
    could point `get_gradients` to this function. This is a modified version of
    `tf.keras.optimizers.Optimizer.get_gradients`.

    # Arguments:
        optimizer: a `tf.keras.optimizers.Optimizer object`. The optimizer you are using.
        loss: Scalar tensor to minimize.
        params: List of variables.

    # Returns:
      A gradients tensor.

    # Reference:
        [Yong et al., 2020](https://arxiv.org/abs/2004.01461)
    """

    # We here just provide a modified get_gradients() function since we are trying to just compute the centralized
    # gradients at this stage which can be used in other optimizers.
    grads = []
    for grad in K.gradients(loss, params):
        grad_len = len(grad.shape)
        if grad_len > 1:
            axis = list(range(grad_len - 1))
            grad -= tf.reduce_mean(grad,
                                   axis=axis,
                                   keep_dims=True)
        grads.append(grad)

    if None in grads:
        raise ValueError('An operation has `None` for gradient. '
                         'Please make sure that all of your ops have a '
                         'gradient defined (i.e. are differentiable). '
                         'Common ops without gradient: '
                         'K.argmax, K.round, K.eval.')
    if hasattr(optimizer, 'clipnorm') and optimizer.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [
            tf.keras.optimizers.clip_norm(
                g,
                optimizer.clipnorm,
                norm) for g in grads]
    if hasattr(optimizer, 'clipvalue') and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue)
                 for g in grads]
    return grads


def centralized_gradients_for_optimizer(optimizer):
    """Create a centralized gradients functions for a specified optimizer.

    # Arguments:
        optimizer: a `tf.keras.optimizers.Optimizer object`. The optimizer you are using.

    # Usage:

    ```py
    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    >>> opt.get_gradients = gctf.centralized_gradients_for_optimizer(opt)
    >>> model.compile(optimizer = opt, ...)
    ```
    """

    def get_centralized_gradients_for_optimizer(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    return get_centralized_gradients_for_optimizer


def get_train_steps(num_examples):
    """Determine the number of training steps."""
    if args.train_steps is None:
        train_steps = (num_examples * args.train_epochs //
                       args.train_batch_size + 1)
    else:
        print("You Implement the args training steps")
        train_steps = args.train_steps

    return train_steps


'''
********************************************
Training Configure
********************************************
1. Learning Rate
    + particular implementation : Scale Learning Rate Linearly with Batch_SIZE 
    (Warmup: Learning Implementation, and Cosine Anealing + Linear scaling)
   
    # optional not implement yet
    + Schedule Learning with Constrain-Update during training

'''
# Implementation form SimCLR paper (Linear Scale and Sqrt Scale)
# Debug and Visualization
# Section SimCLR Implementation Learning rate BYOL implementation
# https://colab.research.google.com/drive/1MWgcDAqnB0zZlXz3fHIW0HLKwZOi5UBb?usp=sharing


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule.
    Args:
    Base Learning Rate: is maximum learning Archieve (change with scale applied)
    num_example
    """

    def __init__(self, base_learning_rate, num_examples, args, name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self.args = args
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            args = self.args
            warmup_steps = int(
                round(args.warmup_epochs * self.num_examples //
                      args.train_batch_size))
            if args.learning_rate_scaling == 'linear':
                scaled_lr = self.base_learning_rate * args.train_batch_size / 256.
            elif args.learning_rate_scaling == 'sqrt':
                scaled_lr = self.base_learning_rate * \
                    math.sqrt(args.train_batch_size)
            elif args.learning_rate_scaling == 'no_scale':
                scaled_lr = self.base_learning_rate
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    args.learning_rate_scaling))
            learning_rate = (
                step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            total_steps = get_train_steps(self.num_examples)
            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(
                scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))

            return learning_rate


'''
********************************************
Training Configure
********************************************
2. Optimizer Strategy + Regularization (Weight Decay)
    The optimizer will have three Options
    1. Orginal 
    2. Implmentation with Weight Decay
    3. Implementation with Gradient Centralization
    4  Implementation with Weight Decay and Gradient Centralization 
    ## Optional Consider Clip_Norm strategy
'''


class get_optimizer():
    '''
    The optimizer will have three Options
    1. Orginal 
    2. Implmentation with Weight Decay
    3. Implementation with Gradient Centralization
    4  Implementation with Weight Decay and Gradient Centralization 

    ## Optional Consider Clip_Norm strategy

    '''

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def orignal_optimizer(self, args):
        '''Args
          - arsg.optimizer type + Learning rate
          Return Optimizer
        '''
        if args.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)

        elif args.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=args.momentum,)

        elif args.optimizer == "LARS":
            optimizer = LARS_optimzer(learning_rate=self.learning_rate,
                                      momentum=args.momentum,
                                      exclude_from_weight_decay=['batch_normalization', 'bias',
                                                                 'head_supervised'])
        return optimizer

    def optimizer_weight_decay(self, args):
        '''Args
          -args.optimizer + args.weight_decay
          Return Optimizer with weight Decay 
        '''
        if args.optimizer == "AdamW":
            optimizer = tfa.optimizers.AdamW(
                weight_decay=args.weight_decay, learning_rate=self.learning_rate)
        if args.optimizer == "SGDW":
            optimizer = tfa.optimizers.SGDW(
                weight_decay=args.weight_decay, learning_rate=self.learning_rate)

        if args.optimizer == "LARSW":
            optimizer = LARS_optimzer(learning_rate=self.learning_rate,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay,
                                      exclude_from_weight_decay=['batch_normalization', 'bias',
                                                                 'head_supervised'])
        return optimizer

    def optimizer_gradient_centralization(self, args):
        '''
        Args
        - args.optimizer + Gradient Centralization 
        return Optimizer with Centralization gradient

        '''
        if args.optimizer == 'AdamGC':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)
        if args.optimizer == "SGDGC":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=args.momentum)
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)
        if args.optimizer == "LARSGC":
            optimizer = LARS_optimzer(learning_rate=self.learning_rate,
                                      momentum=args.momentum,
                                      exclude_from_weight_decay=['batch_normalization', 'bias',
                                                                 'head_supervised'])
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)
        return optimizer

    def optimizer_weight_decay_gradient_centralization(self, args):

        if args.optimizer == "AdamW_GC":
            optimizer = tfa.optimizers.AdamW(
                weight_decay=args.weight_decay, learning_rate=self.learning_rate)
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)

        if args.optimizer == "SGDW_GC":
            optimizer = tfa.optimizers.SGDW(
                weight_decay=args.weight_decay, learning_rate=self.learning_rate)
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)

        if args.optimizer == "LARSW_GC":
            optimizer = LARS_optimzer(learning_rate=self.learning_rate,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay,
                                      exclude_from_weight_decay=['batch_normalization', 'bias',
                                                                 'head_supervised'])
            optimizer.get_gradients = centralized_gradients_for_optimizer(
                optimizer)

        return optimizer
