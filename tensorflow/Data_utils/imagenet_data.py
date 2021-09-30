'''
This Implementaion three Different Augmentation Strategy for ImageNet Dataset
1. Baseline - SimCLR Augmentation 
2. RandAug - RandAug Augmentation (Original and Modify)
3. AutoAugment -- Auto Augmentation Policies 

'''


import numpy as np
import tensorflow as tf
# Augmentation Policy
from imutils import paths
import os
import imgaug.augmenters as iaa
from Data_augmentation_policy.Simclr_augment_random import custom_augment
from Data_augmentation_policy.RandAugment import tfa_randaug, tfa_randaug_rand_crop_fliping, tfa_randaug_rand_ditris_uniform_croping
from Data_augmentation_policy.Auto_Augment import tfa_AutoAugment, tfa_AutoAugment_rand_crop_flip, tfa_AutoAugment_rand_distribe_crop_global_local_views_flip
from official.vision.image_classification.augment import RandAugment, AutoAugment

AUTO = tf.data.experimental.AUTOTUNE
SEED = 26


class imagenet_dataset():
    def __init__(self, IMG_SIZE, BATCH_SIZE, img_path):
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.x_train = list(paths.list_images(img_path))

    @classmethod
    def parse_images(self, image_path):
        # Loading and reading Image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        #img=tf.image.convert_image_dtype(img, tf.float32)

        return img

    @classmethod
    def parse_images_label(self, image_path):
        img = tf.io.read_file(image_path)
        # img = tf.image.decode_jpeg(img, channels=3) # decode the image back to proper format
        img = tf.io.decode_jpeg(img, channels=3)
        label = tf.strings.split(image_path, os.path.sep)[3]
        # print(label)
        return img, label

    def ssl_Simclr_Augment_policy(self):

        train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x: (custom_augment(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x: (custom_augment(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        return train_ds

    def ssl_Randaug_Augment_IMGAUG_policy(self, num_transform, magnitude, crop_size, min_scale, max_scale, high_resol=True,  mode="original"):

        if crop_size is None:
            raise ValueError("you input invalide crop_size")

        rand_aug_apply = iaa.RandAugment(n=num_transform, m=magnitude)

        def imgaug_randaug(images):
            '''
            Args:
            images: A batch tensor [batch, with, height, channels]
            rand_aug: a function to apply Random transformation 
            Return: 
            Images: A batch of Applied transformation [batch, with, height, channels]
            '''

            # Input to `augment()` is a TensorFlow tensor which
            # is not supported by `imgaug`. This is why we first
            # convert it to its `numpy` variant.
            images = tf.cast(images, tf.uint8)
            images = rand_aug_apply(images=images.numpy())
            #images = (images.astype(np.float32))/255.
            images = tf.cast(images, tf.float32)/255.
            return images

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

        def rand_flip_crop_global_local_view(image, min_scale, max_scale, crop_size, high_resol=True):

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

        if mode == "orginal":
            print(" You Implement Imgaug Original")

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform=num_transform, magnitude=magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform, magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "crop":

            print(" You implement Croping with ImgAug")

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)

                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (flip_random_crop(x, crop_size)),  num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform=num_transform, magnitude=magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (flip_random_crop(x, crop_size)),  num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform, magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "global_local_crop":

            print("You implement Global and Local Crop View")

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)

                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (rand_flip_crop_global_local_view(x, min_scale, max_scale, crop_size, high_resol=high_resol)),  num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform=num_transform, magnitude=magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (rand_flip_crop_global_local_view(x, min_scale, max_scale, crop_size, high_resol=high_resol)),  num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            # .map(lambda x: (imgaug_randaug(x, num_transform, magnitude)), num_parallel_calls=AUTO)
                            .map(lambda x: (tf.py_function(imgaug_randaug,  [x], [tf.float32])[0]), num_parallel_calls=AUTO)
                            .prefetch(AUTO)
                            )

        return train_ds

    def ssl_Auto_Augment_TFA_policy(self, crop_size, min_scale, max_scale, high_resol=True, mode="original"):

        if crop_size is None:
            raise ValueError("you enter invalid crop_size")
        #mode ["original", "crop", "global_local_crop"]
        if high_resol:
            print("You Implement the Global Views")
        else:
            print("you implement local views")

        if mode == "original":
            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment(x,)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train))
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment(x,)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "crop":

            print("implement AutoAugment Rand Croping Fliping")
            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment_rand_crop_flip(x,  crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment_rand_crop_flip(x, crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "global_local_crop":

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                              .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                  num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment_rand_distribe_crop_global_local_views_flip(x,  crop_size, min_scale, max_scale, high_resol=high_resol)), num_parallel_calls=AUTO)
                            # .map(lambda x: (tfa_AutoAugment_rand_crop_flip(x,  crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                              .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                  num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_AutoAugment_rand_distribe_crop_global_local_views_flip(x, crop_size, min_scale, max_scale, high_resol=high_resol)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        else:
            raise ValueError("Implementation mode is node in design")

        return train_ds

    def ssl_RandAugment_TFA_policy(self, num_transform, magnitude, crop_size, min_scale, max_scale, high_resol=True, mode="original"):

        if crop_size is None:
            raise ValueError("you enter invalid crop_size")
        #mode ["original", "crop", "global_local_crop"]
        if high_resol:
            print("You Implement the Global Views")
        else:
            print("you implement local views")

        if mode == "original":
            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug(x, num_transform, magnitude,)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug(x, num_transform, magnitude,)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "crop":

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug_rand_crop_fliping(x, num_transform, magnitude, crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug_rand_crop_fliping(x, num_transform, magnitude, crop_size)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        elif mode == "global_local_crop":

            train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug_rand_ditris_uniform_croping(x, num_transform, magnitude,  crop_size, min_scale, max_scale, high_resol=high_resol)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                            .shuffle(self.BATCH_SIZE * 100, seed=SEED)
                            .map(self.parse_images,  num_parallel_calls=AUTO)
                            .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                 num_parallel_calls=AUTO,
                                 )
                            .map(lambda x: (tfa_randaug_rand_ditris_uniform_croping(x, num_transform, magnitude, crop_size, min_scale, max_scale, high_resol=high_resol)), num_parallel_calls=AUTO)
                            .batch(self.BATCH_SIZE)
                            .prefetch(AUTO)
                            )

            train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

        else:
            raise ValueError("Implementation mode is node in design")

        return train_ds
