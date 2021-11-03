import tensorflow as tf
import os 
from imutils import paths
from Augmentation_strategy.byol_simclr_multi_croping_augmentation import simclr_augment_randcrop_global_views, simclr_augment_inception_style
from official.vision.image_classification.augment import RandAugment

AUTO = tf.data.experimental.AUTOTUNE


def supervised_augmentation(image): 
    '''
    Args:
        images: A batch tensor [batch, with, height, channels]
        rand_aug: a function to apply Random transformation 
    Return: 
        Images: A batch of Applied transformation [batch, with, height, channels]
    '''
    rand_aug_apply = RandAugment(n=1, m=7)

    image = rand_aug_apply.distort(image)
    image = tf.cast(image, dtype=tf.float32)/255.

    return image


class imagenet_dataset():

    def __init__(self, IMG_SIZE, BATCH_SIZE,SEED, img_path, val_path=None):
        '''
        args: 
        IMG_SIZE: Image training size 
        BATCH_SIZE: Distributed Batch_size for training multi-GPUs

        image_path: Directory to train data 
        val_path:   Directory to validation or testing data
        
        '''
        
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.seed= SEED
        self.val_train= list(paths.list_images(val_path))
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
    
    
    def supervised_validation(self, input_context): 
        '''This for Supervised validation training'''
        dis_tributed_batch=input_context.get_per_replica_batch_size(self.BATCH_SIZE)
        option= tf.data.Options()
        option.experimental_distribute.auto_shard_policy= tf.data.experimental.AutoShardPolicy.DATA

        val_ds = (tf.data.Dataset.from_tensor_slices(self.val_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images_label,  num_parallel_calls=AUTO)

                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                                num_parallel_calls=AUTO,)
                        .map(lambda x, y: (supervised_augmentation(x), y), num_parallel_calls=AUTO)
                        #.batch(self.BATCH_SIZE)
                        #.prefetch(AUTO)
                        ) 
        val_ds.with_options(option)
        val_ds = val_ds.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)
        val_ds = val_ds.batch(dis_tributed_batch)
        # 2. modify dataset with prefetch
        val_ds = val_ds.prefetch(AUTO)

        return val_ds

   
    def simclr_inception_style_crop(self, input_context):
        '''
        This class property return self-supervised training data
        '''
        dis_tributed_batch=input_context.get_per_replica_batch_size(self.BATCH_SIZE)
        option= tf.data.Options()
        option.experimental_distribute.auto_shard_policy= tf.data.experimental.AutoShardPolicy.DATA

        train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                num_parallel_calls=AUTO,
                                )
                        .map(lambda x: (simclr_augment_inception_style(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        #.batch(self.BATCH_SIZE)
                        #.prefetch(AUTO)
                        )
        train_ds_one.with_options(option)

        train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                num_parallel_calls=AUTO,
                                )
                        .map(lambda x: (simclr_augment_inception_style(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        #.batch(self.BATCH_SIZE)
                        #.prefetch(AUTO)
                        )
        train_ds_two.with_options(option)


        train_ds= tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds = train_ds.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)
        train_ds = train_ds.batch(dis_tributed_batch)
        # 2. modify dataset with prefetch
        train_ds = train_ds.prefetch(AUTO)

        return train_ds
        
  
    def simclr_random_global_crop(self, input_context):
        '''
            This class property return self-supervised training data
        '''
        dis_tributed_batch=input_context.get_per_replica_batch_size(self.BATCH_SIZE)

        option= tf.data.Options()
        option.experimental_distribute.auto_shard_policy= tf.data.experimental.AutoShardPolicy.DATA

        train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                num_parallel_calls=AUTO,
                                )
                        .map(lambda x: (simclr_augment_randcrop_global_views(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        #.batch(self.BATCH_SIZE)
                        #.prefetch(AUTO)
                        )
        train_ds_one.with_options(option)

        train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                                num_parallel_calls=AUTO,
                                )
                        .map(lambda x: (simclr_augment_randcrop_global_views(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        #.batch(self.BATCH_SIZE)
                        #.prefetch(AUTO)
                        )
        train_ds_two.with_options(option)


        train_ds= tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds = train_ds.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)
        train_ds = train_ds.batch(dis_tributed_batch)
        # 2. modify dataset with prefetch
        train_ds = train_ds.prefetch(AUTO)

        return train_ds

    