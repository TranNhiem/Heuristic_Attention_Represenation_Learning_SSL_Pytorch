'''
Auto Augmentation Policy Focus on Object Detection Task 
#Reference Implementation on Object Detection with 
1. AutoAugment 3 Policies (V0- V3)
 Barret, et al. Learning Data Augmentation Strategies for Object Detection.
    Arxiv: https://arxiv.org/abs/1906.11172
2. RandomAugment --> Also apply for object Detection Models

## Reference GitHub for implementation
[1] https://github.com/google/automl/blob/master/efficientdet/aug/autoaugment.py
[2] https://github.com/tensorflow/models/blob/master/official/vision/image_classification/augment.py
'''
# Current implementation will Deploy for Images WITHOUT BOX
import tensorflow as tf
from official.vision.image_classification.augment import AutoAugment

'''
AutoAugment Policy V0-- implementation 
  V0-->   policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]

  Policy_simple -->  = [[('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
    ]

'''


def tfa_AutoAugment(image):
    '''
    Args:
     image: A tensor [ with, height, channels]
     AutoAugment: a function to apply Policy transformation [v0, policy_simple]
    Return: 
      Image: A tensor of Applied transformation [with, height, channels]
    '''
    '''Version 1  AutoAugmentation'''
    augmenter_apply = AutoAugment(augmentation_name='v0')
    image = augmenter_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)/255.
    return image


def tfa_AutoAugment_rand_crop_flip(image, crop_size):
    '''
    Args: 
       image: A tensor [ with, height, channels]
       crop_size: Apply Random Crop_Flip Image before Apply AutoAugment
       AutoAugment: a function to apply Policy transformation [v0, policy_simple]

      Return: 
        Image: A tensor of Applied transformation [with, height, channels]
    '''

    '''Version 2 Auto Augmentation'''
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (crop_size, crop_size, 3))

    augmenter_apply = AutoAugment(augmentation_name='v0')
    image = augmenter_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)/255.
    return image


def tfa_AutoAugment_rand_distribe_crop_global_local_views_flip(image, crop_size, min_scale, max_scale, high_resol=True):
    '''
      Args:
       image: A tensor [ with, height, channels]
       crop_size: Rand --> Flipping --> random_distribute_uniform (min_scale, max_scale) 
       high_resol --> True: For Global crop_view, False: For Local crop views
       AutoAugment: a function to apply AutoAugment transformation 

      Return: 
        Image: A tensor of Applied transformation [with, height, channels]
    '''
    '''Version 1 RandAug Augmentation'''
    image = tf.image.random_flip_left_right(image)
    if high_resol:
        image_shape = tf.cast((crop_size * 1.4), dtype=tf.int32)
        image_shape = tf.cast(image_shape, tf.float32)
        # print(image_shape)
        image = tf.image.resize(image, (image_shape, image_shape))
    else:
        image_shape = tf.cast(crop_size * 0.8, dtype=tf.int32)
        image_shape = tf.cast(image_shape, tf.float32)
        # print(image_shape)
        image = tf.image.resize(image, (image_shape, image_shape))
    size = tf.random.uniform(shape=(
        1,), minval=min_scale*image_shape, maxval=max_scale*image_shape, dtype=tf.float32)
    size = tf.cast(size, tf.int32)[0]
    # Get crop_size
    crop = tf.image.random_crop(image, (size, size, 3))
    # Return image with Crop_size
    image = tf.image.resize(crop, (crop_size, crop_size))

    augmenter_apply = AutoAugment(augmentation_name="v0")
    image = augmenter_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)/255.

    return image
