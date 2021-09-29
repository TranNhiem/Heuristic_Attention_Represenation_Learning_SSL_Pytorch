'''
RandAug implementation for Self-Supervised Learning
paper: https://arxiv.org/abs/1909.13719
RandAug with 2 Version
Version 1 from Original Paper (14 Transformations )
Version 2 Modify with Adding Multiple Transformation (22 transformation from Imgaug API)
'''

import imgaug.augmenters as iaa
from official.vision.image_classification.augment import RandAugment
import tensorflow as tf
import numpy as np

'''
RandAug Augmentation
  available_ops = [
          'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
          'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
          'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']
'''
# Aplly Original


def tfa_randaug(image, num_transform, magnitude):
    '''
    Args:
     image: A tensor [ with, height, channels]
     RandAugment: a function to apply Random transformation
    Return:
      Image: A tensor of Applied transformation [with, height, channels]
    '''
    '''Version 1 RandAug Augmentation'''
    augmenter_apply = RandAugment(
        num_layers=num_transform, magnitude=magnitude)
    image = augmenter_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)

    return image

# Aplly Stragey with default Croping and Flipping


def tfa_randaug_rand_crop_fliping(image, num_transform, magnitude, crop_size):
    '''
    Args:
     image: A tensor [ with, height, channels]
     crop_size: for random Flip--> crop_size of Image
     RandAugment: a function to apply Random transformation
    Return:
      Image: A tensor of Applied transformation [with, height, channels]
    '''

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (crop_size, crop_size, 3))

    augmenter_apply = RandAugment(
        num_layers=num_transform, magnitude=magnitude)
    image = augmenter_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)/255.

    return image

# Apply Strategy RandAug base on Image Resolution by Croping


def tfa_randaug_rand_ditris_uniform_croping(image, num_transform, magnitude, crop_size, min_scale, max_scale, high_resol=True):
    '''
    Args:
     image: A tensor [ with, height, channels]
     crop_size: random crop_size of Image (base_one, Min-Max Scale, with Random_uniform_distri)
     high_resol: Aim for Croping the Image at Global-- Local Views (True- Global views, False Local Views)
     RandAugment: a function to apply Random transformation
    Return:
      Image: A tensor of Applied transformation [with, height, channels]
    '''

    image = tf.image.random_flip_left_right(image)
    if high_resol:
        image_shape = tf.cast((crop_size * 1.4), dtype=tf.int32)
        image_shape = tf.cast(image_shape, tf.float32)
        image = tf.image.resize(image, (image_shape, image_shape))
    else:
        image_shape = tf.cast(crop_size * 0.8, dtype=tf.int32)
        image_shape = tf.cast(image_shape, tf.float32)
        image = tf.image.resize(image, (image_shape, image_shape))

    size = tf.random.uniform(shape=(
        1,), minval=min_scale*image_shape, maxval=max_scale*image_shape, dtype=tf.float32)
    size = tf.cast(size, tf.int32)[0]
    image = tf.image.resize(size, (crop_size, crop_size))

    augmenter_apply = RandAugment(
        num_layers=num_transform, magnitude=magnitude)
    image = augmenter_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)/255.

    return image


'''Version2  RandAug Augmentation'''
# rand_aug = iaa.RandAugment(n=2, m=7)


def imgaug_randaug(images, num_transform, magnitude):
    '''
    Args:
     images: A batch tensor [batch, with, height, channels]
     rand_aug: a function to apply Random transformation 
    Return: 
      Images: A batch of Applied transformation [batch, with, height, channels]
    '''
    rand_aug_apply = iaa.RandAugment(n=num_transform, m=magnitude)

    # Input to `augment()` is a TensorFlow tensor which
    # is not supported by `imgaug`. This is why we first
    # convert it to its `numpy` variant.
    images = tf.cast(images, tf.uint8)
    images = rand_aug_apply(images=images.numpy())
    # images = (images.astype(np.float32))/255.
    images = tf.cast(images, tf.float32)/255.
    return images
