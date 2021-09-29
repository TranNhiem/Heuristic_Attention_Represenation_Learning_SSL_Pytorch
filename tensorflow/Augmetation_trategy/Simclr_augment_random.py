'''
Implementation SimCLR Augmentation Augmentation Transformation Policy for BYOL paper 
SimCLR paper: https://arxiv.org/abs/2002.05709
BYOL paper: https://arxiv.org/pdf/2006.07733.pdf
'''
# Reference
import tensorflow as tf

# This random brightness implement in SimCLRV2

# Probability implement and Intensity inherence from BYOL paper


def color_jitter(image, strength=[0.4, 0.4, 0.2, 0.1]):
    '''
    Args: 
      image: tensor shape of [height, width, channels]
      strength: lists of intensity for color distortion
    Return 
      A distortion transofrm tensor[height, width, channels]
    '''
    x = tf.image.random_brightness(image, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1-0.8 * strength[1], upper=1 + 0.8 * strength[1])
    x = tf.image.random_saturation(
        x, lower=1-0.8*strength[2], upper=1 + 0.8 * strength[2])
    x = tf.image.random_hue(x, max_delta=0.2*strength[3])
    # Color distor transform can disturb the natural range of RGB -> Hence ->clib by value
    x = tf.clip_by_value(x, 0, 255)
    return x
# Alternative random_crop (Simclr_github)


def flip_random_crop(image, crop_size):
    '''
    Args: 
      image: tensor shape of [height, width, channels]
      crop_size: using for random crop 
    Return: 
      A tensor transform with Flipping and Crop same_size as image if Crop==img_size
    '''
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (crop_size, crop_size, 3))
    return image


def color_drop(image):
    '''
    Args: 
      image: Tensor shape of [Height, width, channels]
    Return:
      A convert RGB-> gray transform 
    '''
    x = tf.image.rgb_to_grayscale(image)
    x = tf.tile(x, [1, 1, 3])
    return x

# This section for Random Blur


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
    """Blurs the given image with separable convolution.
    Args:
      image: Tensor of shape [height, width, channels] and dtype float to blur.
      kernel_size: Integer Tensor for the size of the blur kernel. This is should
        be an odd number. If it is an even number, the actual kernel size will be
        size + 1.
      sigma: Sigma value for gaussian operator.
      padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
    Returns:
      A Tensor representing the blurred image.
    """
    radius = tf.cast((kernel_size / 2), dtype=tf.int32)
    kernel_size = radius * 2 + 1
    x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
    blur_filter = tf.exp(
        -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
        # Tensorflow requires batched input to convolutions, which we can fake with
        # an extra dimension.
        image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
        blurred = tf.squeeze(blurred, axis=0)
    return blurred

# Attention we also can implement Batches


def random_blur(image, ):  # IMG_SIZE
    '''
    Args: 
      Image: A tensor [height, width, channels]
      IMG_SIZE: image_size 
      p: probability of applying transformation 
    Returns: 
      A image tensor that Blur
    '''
    # finding image using shape of tensor
    IMG_SIZE = image.shape[1]
    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    image_blur = gaussian_blur(
        image, kernel_size=IMG_SIZE // 10, sigma=sigma, padding='SAME')
    return image_blur


def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x

# In Implementation the Crop_size should equal to IMG_SIZE or smaller


def custom_augment(image, crop_size):
    # IMG_SIZE=IMG_SIZE
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.
    image = flip_random_crop(image, crop_size)
    image = random_apply(color_jitter, p=0.8, x=image, )
    image = random_apply(color_drop, p=0.2, x=image, )
    image = random_apply(random_blur, p=1.0, x=image,)
    image = tf.cast(image, tf.float32)/255.
    return image
