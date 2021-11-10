import os
from absl import flags
import tensorflow as tf
from imutils import paths
from byol_simclr_multi_croping_augmentation import simclr_augment_randcrop_global_views, simclr_augment_inception_style, supervised_augment_eval, simclr_augment_randcrop_global_view_image_mask, simclr_augment_inception_style_image_mask
from absl import logging
import numpy as np
import random
AUTO = tf.data.experimental.AUTOTUNE

FLAGS = flags.FLAGS


class imagenet_dataset_single_machine():

    def __init__(self, img_size, train_batch, val_batch, strategy, img_path=None, x_val=None, x_train=None,bi_mask=True):
        '''
        args: 
        IMG_SIZE: Image training size 
        BATCH_SIZE: Distributed Batch_size for training multi-GPUs

        image_path: Directory to train data 
        val_path:   Directory to validation or testing data

        '''
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = train_batch
        self.val_batch = val_batch
        self.strategy= strategy

        self.seed = FLAGS.SEED
        self.x_train = x_train
        self.x_val = x_val

        if bi_mask:
            self.bi_mask=[]
            for p in self.x_train:
                self.bi_mask.append(path.replace("1K/", "binary_image_by_USS/").replace("JPEG","png"))
       
        ## Path for loading all Images 
        # For training 
        all_train_class = []
        for image_path in x_train:
            # label = tf.strings.split(image_path, os.path.sep)[5]
            # all_train_class.append(label.numpy())
            label= image_path.split("/")[5]
            all_train_class.append(label)

        number_class = set(all_train_class)
        all_cls =list(number_class)
        
        class_dic = dict()
        for i in range(999):
            class_dic[all_cls[i]] = i+1
        
        numeric_train_cls = []
        for i in range(len(all_train_class)):
            for k, v in class_dic.items():
                if all_train_class[i] == k:
                    numeric_train_cls.append(v)

        ## For Validation
        all_val_class = []
        for image_path in x_val:
            label= image_path.split("/")[5]
            all_val_class.append(label)

    
        numeric_val_cls = []
        for i in range(len(all_val_class)):
            for k, v in class_dic.items():
                if all_train_class[i] == k:
                    numeric_val_cls.append(v)

        self.x_train_lable = tf.one_hot(numeric_train_cls, depth=999)
        self.x_val_lable = tf.one_hot(numeric_val_cls, depth=999)


        if img_path is not None:
            dataset = list(paths.list_images(img_path))
            self.dataset_shuffle = random.sample(dataset, len(dataset))
            self.x_val = self.dataset_shuffle[0:50000]
            self.x_train = self.dataset_shuffle[50000:]

        if self.bi_mask is not None: 
            self.x_train_image_mask= zip(self.x_train, self.bi_mask)

    @classmethod
    def parse_images(self, image_path):
        # Loading and reading Image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img=tf.image.convert_image_dtype(img, tf.float32)
        return img

    @classmethod
    def parse_images_lable_pair(self, image_path, lable):
        # Loading and reading Image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img=tf.image.convert_image_dtype(img, tf.float32)
        
        return img, lable
    
    @classmethod
    def parse_images_mask_lable_pair(self, image_mask_path, lable):
        # Loading and reading Image
        image_path, mask_path= image_mask_path[0], image_mask_path[1] 
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img=tf.image.convert_image_dtype(img, tf.float32)
        img= tf.image.resize(img, (self.IMG_SIZE, self.IMG_SIZE)

        bi_mask = tf.io.read_file(mask_path)
        bi_mask = tf.io.decode_jpeg(bi_mask, channels=1)
        bi_mask= tf.image.resize(bi_mask, (self.IMG_SIZE,self.IMG_SIZE)
        return img, bi_mask, lable

    @classmethod
    def parse_images_label(self, image_path):
        img = tf.io.read_file(image_path)
        # img = tf.image.decode_jpeg(img, channels=3) # decode the image back to proper format
        img = tf.io.decode_jpeg(img, channels=3)
        label = tf.strings.split(image_path, os.path.sep)[4]
        # print(label)
        return img, label

    def supervised_validation(self):
        '''This for Supervised validation training'''

        val_ds = (tf.data.Dataset.from_tensor_slices((self.x_val, self.x_val_lable))
                  .shuffle(self.val_batch * 100, seed=self.seed)
                  .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)

                  .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                       num_parallel_calls=AUTO,)
                  .map(lambda x, y: (supervised_augment_eval(x, FLAGS.IMG_height, FLAGS.IMG_width, FLAGS.randaug_transform, FLAGS.randaug_magnitude), y), num_parallel_calls=AUTO)
                  .batch(self.BATCH_SIZE)
                  .prefetch(AUTO)
                  )

        val_ds= self.strategy.experimental_distribute_dataset(val_ds)

        return val_ds

    def simclr_inception_style_crop(self):
        '''
        This class property return self-supervised training data
        '''
        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        #.map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x, y: (simclr_augment_inception_style(x, self.IMG_SIZE), y), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        #train_ds_one= self.strategy.experimental_distribute_dataset(train_ds_one)

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        #.map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x, y: (simclr_augment_inception_style(x, self.IMG_SIZE), y), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        #train_ds_one= self.strategy.experimental_distribute_dataset(train_ds_two)
        
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds= self.strategy.experimental_distribute_dataset(train_ds)
        # train_ds = train_ds.batch(self.BATCH_SIZE)
        # # 2. modify dataset with prefetch
        # train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def simclr_random_global_crop(self):
  
        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        #.map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x, y: (simclr_augment_randcrop_global_views(x, self.IMG_SIZE), y), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        #.map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)
                        .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x, y: (simclr_augment_randcrop_global_views(x, self.IMG_SIZE), y), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        #adding the distribute data to GPUs
        train_ds= self.strategy.experimental_distribute_dataset(train_ds)

        return train_ds

    def simclr_inception_style_crop_image_mask(self):
        
        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        #.map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)
                        .map(lambda x, y,z: (simclr_augment_inception_style_image_mask(x,y, self.IMG_SIZE), z), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        #.map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)
                        .map(lambda x, y,z: (simclr_augment_inception_style_image_mask(x,y, self.IMG_SIZE), z), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        #train_ds_one= self.strategy.experimental_distribute_dataset(train_ds_two)
        
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds= self.strategy.experimental_distribute_dataset(train_ds)
        # train_ds = train_ds.batch(self.BATCH_SIZE)
        # # 2. modify dataset with prefetch
        # train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def simclr_random_global_crop_image_mask(self):
        
        train_ds_one = (tf.data.Dataset.from_tensor_slices((self.x_train_image_mask, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        #.map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)
                        .map(lambda x, y,z: (simclr_augment_randcrop_global_view_image_mask(x,y, self.IMG_SIZE), z), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        

        train_ds_two = (tf.data.Dataset.from_tensor_slices((self.x_train_image_mask, self.x_train_lable))
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        #.map(self.parse_images_label,  num_parallel_calls=AUTO)
                        .map(lambda x, y: (self.parse_images_lable_pair(x, y)),  num_parallel_calls=AUTO)
                        .map(lambda x, y,z: (simclr_augment_randcrop_global_view_image_mask(x,y, self.IMG_SIZE), z), num_parallel_calls=AUTO)
                        .batch(self.BATCH_SIZE)
                        .prefetch(AUTO)
                        )
        #train_ds_one= self.strategy.experimental_distribute_dataset(train_ds_two)
        
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds= self.strategy.experimental_distribute_dataset(train_ds)

class imagenet_dataset_multi_machine():

    def __init__(self, IMG_SIZE, BATCH_SIZE, img_path=None, x_val=None, x_train=None):
        '''
        args: 
        IMG_SIZE: Image training size 
        BATCH_SIZE: Distributed Batch_size for training multi-GPUs

        image_path: Directory to train data 
        val_path:   Directory to validation or testing data

        '''
        self.IMG_SIZE = IMG_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.seed = FLAGS.SEED
        self.x_train = x_train
        self.x_val = x_val

        if img_path is not None:
            dataset = list(paths.list_images(img_path))
            self.dataset_shuffle = random.sample(dataset, len(dataset))
            self.x_val = self.dataset_shuffle[0:50000]
            self.x_train = self.dataset_shuffle[50000:]

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
        dis_tributed_batch = input_context.get_per_replica_batch_size(
            self.BATCH_SIZE)

        logging.info('Global batch size: %d', self.BATCH_SIZE)
        logging.info('Per-replica batch size: %d', dis_tributed_batch)
        logging.info('num_input_pipelines: %d',
                     input_context.num_input_pipelines)
        option = tf.data.Options()
        option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        val_ds = (tf.data.Dataset.from_tensor_slices(self.x_val)
                  .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                  .map(self.parse_images_label,  num_parallel_calls=AUTO)

                  .map(lambda x, y: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE)), y),
                       num_parallel_calls=AUTO,)
                  .map(lambda x, y: (supervised_augment_eval(x, FLAGS.IMG_height, FLAGS.IMG_width, FLAGS.randaug_transform, FLAGS.randaug_magnitude), y), num_parallel_calls=AUTO)
                  # .batch(self.BATCH_SIZE)
                  # .prefetch(AUTO)
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
        dis_tributed_batch = input_context.get_per_replica_batch_size(
            self.BATCH_SIZE)

        logging.info('Global batch size: %d', self.BATCH_SIZE)
        logging.info('Per-replica batch size: %d', dis_tributed_batch)
        logging.info('num_input_pipelines: %d',
                     input_context.num_input_pipelines)
        option = tf.data.Options()
        option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x: (simclr_augment_inception_style(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        # .batch(self.BATCH_SIZE)
                        # .prefetch(AUTO)
                        )
        train_ds_one.with_options(option)

        train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x: (simclr_augment_inception_style(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        # .batch(self.BATCH_SIZE)
                        # .prefetch(AUTO)
                        )
        train_ds_two.with_options(option)

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
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

        dis_tributed_batch = input_context.get_per_replica_batch_size(
            self.BATCH_SIZE)
        logging.info('Global batch size: %d', self.BATCH_SIZE)
        logging.info('Per-replica batch size: %d', dis_tributed_batch)
        logging.info('num_input_pipelines: %d',
                     input_context.num_input_pipelines)

        option = tf.data.Options()

        option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        train_ds_one = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x: (simclr_augment_randcrop_global_views(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        # .batch(self.BATCH_SIZE)
                        # .prefetch(AUTO)
                        )
        train_ds_one.with_options(option)

        train_ds_two = (tf.data.Dataset.from_tensor_slices(self.x_train)
                        .shuffle(self.BATCH_SIZE * 100, seed=self.seed)
                        .map(self.parse_images,  num_parallel_calls=AUTO)
                        .map(lambda x: (tf.image.resize(x, (self.IMG_SIZE, self.IMG_SIZE))),
                             num_parallel_calls=AUTO,
                             )
                        .map(lambda x: (simclr_augment_randcrop_global_views(x, self.IMG_SIZE)), num_parallel_calls=AUTO)
                        # .batch(self.BATCH_SIZE)
                        # .prefetch(AUTO)
                        )
        train_ds_two.with_options(option)

        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        train_ds = train_ds.shard(input_context.num_input_pipelines,
                                  input_context.input_pipeline_id)
        train_ds = train_ds.batch(dis_tributed_batch)
        # 2. modify dataset with prefetch
        train_ds = train_ds.prefetch(AUTO)

        return train_ds

    def simclr_inception_style_crop_image_mask(self, input_context):
        pass

    def simclr_random_global_crop_image_mask(self, input_context):
        pass
