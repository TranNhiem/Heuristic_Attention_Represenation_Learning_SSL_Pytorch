'''

Implementation of Convolution with Perciever Transformer Architecture

'''



import tensorflow as tf
import numpy as np
from tensorflow.python.autograph.operators.py_builtins import len_
from tensorflow.python.keras.engine.base_layer import InputSpec
import matplotlib.pyplot as plt

## from position_encoding_type import FourierPositionEncoding
####################################################################################
'''DEFINE Hyperparameter Unroll the Image'''
####################################################################################
# input_shape = (32, 32, 3)
# # Try to keep latten array small
# IMG_SIZE = 32
# num_class = 100
# patch_size = 2
# num_patches = (IMG_SIZE//patch_size)**2

# latten_dim = 256  # size of latten array --> (N)
# # Embedding output of Data PATCHES + LATTEN array --> Project DIM --> D
# projection_dim = 256

# # Learnable array
# # (NxD) #--> OUTPUT( [Q, K][Conetent information, positional])
# latten_array = latten_dim * projection_dim

# NUM_TRANSFORM_HEAD = 8  # --> Each Attention Module
# # Encoder -- Decoder are # --> Increasing block create deeper Transformer model
# NUM_TRANSFORM_BLOCK = 4

# # Corresponding with Depth and devided between number of trasnform_block
# NUM_LAYER_CROSS_TRANSFORMER = 4

# # 2 layer MLP Dense with number of Unit= pro_dim
# feed_forward = [projection_dim, projection_dim]
# classification_head = [projection_dim, num_class]

# print(f"Image size: {IMG_SIZE} X {IMG_SIZE} = {IMG_SIZE ** 2}")
# print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
# print(f"Patches per image: {num_patches}")
# print(
#     f"Elements per patch [patch_size*patch_size] (3 channels RGB): {(patch_size ** 2) * 3}")
# print(f"Latent array shape: {latten_dim} X {projection_dim}")
# print(f"Data array shape: {num_patches} X {projection_dim}")


#####################################################################################
'''Create Feed Forward Network'''
######################################################################################
# Feed forward network contain in Attention all you need 2 linear and 1 ReLu in middel


def create_ffn(units_neuron, dropout_rate):
    '''
    args: Layers_number_neuron  == units_neuron
        example units_neuron=[512, 256, 256] --> layers=len(units_neuron), units= values of element inside list
    dropout rate--> adding 1 dropout percentages layer Last ffn model

    return  FFN model in keras Sequential model
    '''
    ffn_layers = []
    for units in units_neuron[:-1]:
        ffn_layers.append(tf.keras.layers.Dense(
            units=units, activation=tf.nn.gelu))

    ffn_layers.append(tf.keras.layers.Dense(units=units_neuron[-1]))
    ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))
    ffn = tf.keras.Sequential(ffn_layers)
    return ffn


def create_classification_ffn(units_neuron, dropout_rate):
    '''
    args: Layers_number_neuron  == units_neuron
        example units_neuron=[512, 256, 256] --> layers=len(units_neuron), units= values of element inside list
    dropout rate--> adding 1 dropout percentages layer Last ffn model

    return  FFN model in keras Sequential model
    '''
    ffn_layers = []
    for units in units_neuron[:-1]:
        ffn_layers.append(tf.keras.layers.Dense(
            units=units, activation=tf.nn.gelu))

    ffn_layers.append(tf.keras.layers.Dense(
        units=units_neuron[-1], activation='softmax'))
    # ffn_layers.append(tf.keras.layers.Dropout(dropout_rate))

    ffn = tf.keras.Sequential(ffn_layers)

    return ffn


####################################################################################
'''Extract Patches Unroll the Image'''
####################################################################################

# Patches Extract


class patches(tf.keras.layers.Layer):
    '''
    args: Patch_size the size of crop you expect to Unroll image into sequences

    return the total number patches
    '''

    def __init__(self, patch_size):
        super(patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        # print(patches.shape)
        return patches

# Display function show the image patches Extraction


def display_pathches(image, IMG_SIZE, patch_size):
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis("off")

    # Resize image and Convert to Tensor
    image_resize = tf.image.resize(
        tf.convert_to_tensor([image]), size=(IMG_SIZE, IMG_SIZE))
    # Unroll image to many patches
    patches_unroll = patches(patch_size, )(image_resize)
    # Information of Unroll IMAGE Corresponding with Patch
    print(f'IMG_SIZE: {IMG_SIZE} x {IMG_SIZE}')
    print(f'Implement Patch_size: {patch_size} x {patch_size}')
    print(f'Patches per image: {patches_unroll.shape[1]}')
    print(f'Elements per Patch: {patches_unroll.shape[-1]}')

    # number of rows and columns
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches_unroll[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")
# Custom layer ConvNet unroll patches


class conv_unroll_patches(tf.keras.layers.Layer):
    '''
    Args, 
    Number of Conv Layer
    Spatial_dim_to_projection_dim

    return the PATCHES -- Sequences of patches corresponding with ConvKernel_size
    '''

    def __init__(self, num_conv_layer, spatial2_projcetion_dim, kernel_size=3, stride=1, padding=1,  pooling_kernel_size=3, pooling_stride=2, ):
        super(conv_unroll_patches, self).__init__()
        self.num_conv_layer = num_conv_layer
        self.spatial2_projection_dim = spatial2_projcetion_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride

        # This is our tokenizer.
        self.conv_model = tf.keras.Sequential()
        for i in range(num_conv_layer):
            self.conv_model.add(
                tf.keras.layers.Conv2D(
                    spatial2_projcetion_dim[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(tf.keras.layers.ZeroPadding2D(padding))
            self.conv_model.add(
                tf.keras.layers.MaxPool2D(
                    pooling_kernel_size, pooling_stride, "same")
            )

    def call(self, inputs):
        outputs = self.conv_model(inputs)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        flatten_sequences = tf.reshape(
            outputs,
            (-1, tf.shape(outputs)[1] * tf.shape(outputs)
             [2], tf.shape(outputs)[-1]),
        )
        flatten_sequences = tf.cast(flatten_sequences, dtype=tf.float32)
        return flatten_sequences


####################################################################################
'''Patches Position Encoding '''

# 1 tf.keras.layer.embedding Position Encoding
# LINEAR transform into Vector

# 2 Learnable and Not Learnable Fourier position Encoding
# Multi Freq transforms (Perciever-Perceiver IO)

# 3. Position Sensitive Encoding (Axial DeepLap paper)

# 4. Dinstangle Position Encoding (DeBERTA paper)

####################################################################################
# Encoding Patches [Content and Position]
# 1 LINEAR position encoding tf.keras.layers.Embeddeding
# This position encoding is not learnable


class patch_content_position_encoding(tf.keras.layers.Layer):
    '''
    args:
        num_pathes: number_sequences patches unroll from image
        project-dim; the output of embedding layers: should be the same with latter array
    return
        Embedding position vectors

    '''

    def __init__(self, num_patches, project_dim):  # Noted project_dim == Latten Array

        self.pro_dim = project_dim
        self.num_patches = num_patches
        super(patch_content_position_encoding, self).__init__()
        self.projection = tf.keras.layers.Dense(units=project_dim)  # content
        # LINEAR Position Encoding
        self.position_encoding = tf.keras.layers.Embedding(input_dim=num_patches,
                                                           output_dim=project_dim)  # Position

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoding = self.projection(patches), self.position_encoding(
            positions)
        return encoding

# CUstom layer LInear postiong Encoded


class conv_content_position_encoding_cls(tf.keras.layers.Layer):
    '''
    Building layer to return Position Encoding

    args:
        num_pathes: number_sequences patches unroll from image
        project-dim; the output of embedding layers: should be the same with latter array
    return
        Embedding position vectors

    '''
    ##  num_patches == Sequence_flatten,
    # projection_dim= sequence_length -- depend on the

    # Noted project_dim == Latten Array
    def __init__(self, image_size, num_conv_layer, spatial2projection_dim):
        self.image_size = image_size
        self.patches_sequences_flatten = conv_unroll_patches(
            num_conv_layer, spatial2projection_dim)
        super(conv_content_position_encoding_cls, self).__init__()

    def call(self):  # patches extract from ConvNet
        # LINEAR Position Encoding
        dummy_img_posit = tf.ones(
            (1, self.image_size, self.image_size, 3))
        sequences_flatten = self.patches_sequences_flatten(dummy_img_posit)
        sequences_flatten_out = tf.shape(sequences_flatten)[1]
        projection_dim = tf.shape(sequences_flatten)[-1]
        position_encoding = tf.keras.layers.Embedding(input_dim=sequences_flatten_out,
                                                      output_dim=projection_dim)

        positions = tf.range(start=0, limit=sequences_flatten_out, delta=1)
        position_encoding_out = position_encoding(positions)

        return position_encoding_out

# Custom Function
# This function can work Probaly


def conv_content_position_encoding(image_size, num_conv_layer, spatial2projection_dim):
    '''
    Building layer to return Position Encoding

    args:
        image_size for -> sequence position patches
        num_pathes: number_sequences patches unroll from image
        project-dim; the output of embedding layers: should be the same with latter array
    return
        Embedding position vectors

    '''
    patches_sequence = conv_unroll_patches(
        num_conv_layer, spatial2projection_dim)
    # patches extract from ConvNet
    # LINEAR Position Encoding
    dummy_img_posit = tf.ones(
        (1, image_size, image_size, 3), dtype=tf.float32)
    sequences_flatten = patches_sequence(dummy_img_posit)
    sequences_flatten_out = tf.shape(sequences_flatten)[1]
    projection_dim = tf.shape(sequences_flatten)[-1]
    position_encoding = tf.keras.layers.Embedding(input_dim=sequences_flatten_out,
                                                  output_dim=projection_dim)

    positions = tf.range(start=0, limit=sequences_flatten_out, delta=1)
    position_encoding_out = position_encoding(positions)
    position_encoding_out = tf.cast(
        position_encoding_out, dtype=tf.float32)

    return position_encoding_out


class conv_unroll_patches_position_encoded(tf.keras.layers.Layer):
    '''
    Args, 
    Number of Conv Layer
    Spatial_dim_to_projection_dim

    return the PATCHES -- Sequences of patches corresponding with ConvKernel_size
    '''

    def __init__(self, num_conv_layer, spatial2_projcetion_dim, kernel_size=3, stride=1, padding=1,  pooling_kernel_size=3, pooling_stride=2, ):
        super(conv_unroll_patches_position_encoded, self).__init__()
        self.num_conv_layer = num_conv_layer
        self.spatial2_projcetion_dim = spatial2_projcetion_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride

        # This is our tokenizer.
        conv_model = tf.keras.Sequential()
        for i in range(self.num_conv_layer):
            conv_model.add(
                tf.keras.layers.Conv2D(
                    self.spatial2_projcetion_dim[i],
                    self.kernel_size,
                    self.stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            conv_model.add(tf.keras.layers.ZeroPadding2D(self.padding))
            conv_model.add(tf.keras.layers.MaxPool2D(
                self.pooling_kernel_size, self.pooling_stride, "same")
            )
        self.conv_model = conv_model

    def call(self, inputs):

        outputs = self.conv_model(inputs)

        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        flatten_sequences = tf.reshape(
            outputs,
            (-1, tf.shape(outputs)[1] * tf.shape(outputs)
             [2], tf.shape(outputs)[-1]),
        )
        flatten_sequences = tf.cast(flatten_sequences, dtype=tf.float32)
        return flatten_sequences

    def conv_content_position_encoding(self, image_size):
        '''
        Building layer to return Position Encoding

        args:
            image_size for -> sequence position patches
            num_pathes: number_sequences patches unroll from image
            project-dim; the output of embedding layers: should be the same with latter array
        return
            Embedding position vectors

        '''

        # patches extract from ConvNet
        # LINEAR Position Encoding
        dummy_img_posit = tf.ones(
            (1, image_size, image_size, 3), dtype=tf.float32)
        sequences_flatten = self.call(dummy_img_posit)
        sequences_flatten_out = tf.shape(sequences_flatten)[1]
        projection_dim = tf.shape(sequences_flatten)[-1]
        position_encoding = tf.keras.layers.Embedding(input_dim=sequences_flatten_out,
                                                      output_dim=projection_dim)

        positions = tf.range(start=0, limit=sequences_flatten_out, delta=1)
        position_encoding_out = position_encoding(positions)
        position_encoding_out = tf.cast(
            position_encoding_out, dtype=tf.float32)

        return position_encoding_out, sequences_flatten_out

    def get_config(self):

        configs_item = {
            'num_conv_layer': self.num_conv_layer,
            'spatial2_projection_dim': self.spatial2_projcetion_dim,
            # 'kernel_size': self.kernel_size,
            # 'stride': self.stride,
            # 'padding': self.padding,
            # 'pooling_kernel_size': self.pooling_kernel_size,
            # 'pooling_stride': self.pooling_stride,

        }
        configs = super(conv_unroll_patches_position_encoded,
                        self).get_config()  # .copy()

        return dict(list(configs.items()) + list(configs_item.items()))
        # return config


####################################################################################
'''-----Building Transformer Module----'''
# 1. Convention attention module for encoder --- Decoder

# 2. Cross Attention Module (Perceiver/PerceiverIO)

# 3. Axial Attention Module (Axial DeepLAB paper)

# 4. Distangle Attention Module (DeBERTA paper)
####################################################################################

# 1 Convetion Attention Module


def latten_transformer_attention(lattent_dim, projection_dim, num_multi_head,
                                 num_transformer_block, ffn_units, dropout):
    '''
    Args:
        take lattent_dim as dimension inputs,
        num_multi_head: number multi-head attention for split inputs and feed into
        num_transformer_block: stack attention module multiple time
        ffn_units: number layers, number_units neuron
        dropout: dropout rate at the end of FFN- model
    return
        Attention Encoder model

    '''

    # Input of transformer (N)--> quadratic O(N*N)
    # Input shape=[1, latent_dim, projection_dim]
    inputs = tf.keras.layers.Input(shape=(lattent_dim, projection_dim))
    x0 = inputs
    # stack multiple attention encoder block
    for _ in range(num_transformer_block):
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x0)
        # create multi-head attention
        # attention_axes=None attention over all axes
        multi_head_out = tf.keras.layers.MultiHeadAttention(num_heads=num_multi_head,
                                                            key_dim=projection_dim, dropout=dropout)(x, x)
        # adding skip connection (Between multi head previous layernorm --> Norm again)
        x1 = tf.keras.layers.Add()([multi_head_out, x])
        x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x1)
        # Apply Feed forward network
        x3 = create_ffn(units_neuron=ffn_units, dropout_rate=dropout)
        x3 = x3(x2)

        # Adding skip connection (betwen Layersnorm and ffn_output)
        x0 = tf.keras.layers.Add()([x3, x2])

    # create stack block model
    model = tf.keras.Model(inputs=inputs, outputs=x0)
    return model

# 2 Cross Attention Module


def cross_attention_module(lattent_dim, data_dim, projection_dim, ffn_units, dropout):
    '''
    Args:
        latten_dim: Reduce dimension you expected to
        data_dim: unroll the image num_patchets * projection-dim
        ffn_units: layers len(ffn_units), nurons= value of element inside
        dropout: percentages dropout the last layer FFN
    '''

    # Cross input between Low dimention with high dimention --> low dimention {Language translation}
    inputs = {
        # 1.. Receive the latten array as input shape[1, lattent_dim, projection_dim]
        "latent_array": tf.keras.layers.Input(shape=(lattent_dim, projection_dim)),
        # Receive the data array input shape[batch_size, data_dim, projection_dim]
        # data_dim= num_patches* project_dim
        "data_array": tf.keras.layers.Input(shape=(data_dim, projection_dim))
    }

    # Apply layer norm for each input
    lattent_array = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs["latent_array"])
    data_array = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs["data_array"])

    # Create query tensor: [1, latten_dim, projection_dim]
    query = tf.keras.layers.Dense(units=projection_dim)(lattent_array)
    # Create key tensor: [batch_size, data_dim, projection_dim]
    key = tf.keras.layers.Dense(units=projection_dim)(data_array)
    # Create Value Tensor: [Batch_size, data_dim, projection_dim]
    value = tf.keras.layers.Dense(units=projection_dim)(data_array)

    # Generate cross attention output by multiple Q* V --> Scale/Q_dims--> Softmax *Value
    attention = tf.keras.layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False)
    # adding skip connection 1: between lattent_array and Cross attention output
    attention_output = tf.keras.layers.Add()([attention, lattent_array])

    # Normalize output
    attention_output_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention_output)
    # Apply Feedforward Network
    ffn = create_ffn(units_neuron=ffn_units, dropout_rate=dropout)
    ffn = ffn(attention_output_norm)

    # adding skip connection 2: between attention output and FFN model
    outputs = tf.keras.layers.Add()([ffn, attention_output_norm])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


####################################################################################
'''Perceviver Architecture'''
####################################################################################
# 1. The cross-attention epxect a (lattent_dim, projection_dim) --> latten array
# 2. data array (data_dim, proction_dim) --> data arrray
# ==> Dotproduct (latten_array, data_array) --> (latten_dim, projection_dim)
# Q Generated from lattent array, K, V generated from the encoded image
# data_dim will equal to number of patches after unrol the image.


class perceiver_architecture(tf.keras.Model):

    def __init__(self, patch_size, data_dim, lattent_dim, projection_dim, num_multi_heads,
                 num_transformer_block, ffn_units, dropout, num_model_layer, classifier_units, include_top=False, pooling_mode="1D"):

        self.patch_size = patch_size
        self.data_dim = data_dim  # num_patches* projection_dim
        self.lattent_dim = lattent_dim
        self.projection_dim = projection_dim
        self.num_multi_heads = num_multi_heads
        self.num_transformer_block = num_transformer_block
        self.ffn_units = ffn_units
        self.dropout = dropout
        self.num_model_layer = num_model_layer
        self.classifier_units = classifier_units
        self.include_top = include_top
        self.pooling_mode = pooling_mode
        super(perceiver_architecture, self).__init__(
            name="Perceiver_Architecture")

    def build(self, input_shape):

        # create lattent array with init random values
        self.latent_array = self.add_weight(shape=(self.lattent_dim, self.projection_dim),
                                            initializer="random_normal", trainable=True)

        # create patching module
        self.num_patches = patches(self.patch_size)

        # create patch embedding encoded (position, content information) data input (K, V)
        self.patches_embedding = patch_content_position_encoding(
            self.data_dim, self.projection_dim)

        # Create cross-attention module
        self.cross_attention = cross_attention_module(self.lattent_dim, self.data_dim, self.projection_dim,
                                                      self.ffn_units, self.dropout)

        # Create Latten_transformer_Attention
        self.latent_transformer = latten_transformer_attention(self.lattent_dim, self.projection_dim, self.num_multi_heads,
                                                               self.num_transformer_block, self.ffn_units, self.dropout)

        if self.pooling_mode == "1D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        elif self.pooling_mode == "2D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        else:
            raise Exception("you not implement available pooling mode")

        if self.include_top == True:
            self.classification_head = create_classification_ffn(
                units_neuron=self.classifier_units, dropout_rate=self.dropout)

        super(perceiver_architecture, self).build(input_shape)

    def call(self, inputs):
        # Augmentation option --> self-supervised processing outside
        # augmented= data_augmentation(inputs)
        # this for training using tf.data.Dataset
        # inputs, _ = inputs
        # create patches
        num_patches = self.num_patches(inputs)

        # embedding patches position content information
        embedding_patches = self.patches_embedding(num_patches)

        # passing input to cross attention
        cross_attention_input = {"latent_array": tf.expand_dims(self.latent_array, 0),
                                 "data_array": embedding_patches,
                                 }
        # Apply cross attention --> latent transform --> Stack multiple build deeper model
        for _ in range(self.num_model_layer):
            # Applying cross attention to INPUT
            latent_array = self.cross_attention(cross_attention_input)
            # apply latent attention to cross attention OUTPUT
            latent_array = self.latent_transformer(latent_array)
            # set the latent array out output to the next block
            cross_attention_input["latent_array"] = latent_array

        # Applying Global Average_pooling to generate [Batch_size, projection_dim] representation
        representation = self.global_average_pooling(latent_array)

        if self.include_top == True:
            representation = self.classification_head(representation)

        return representation


class perceiver_architecture_V1(tf.keras.Model):

    def __init__(self, patch_size, data_dim, lattent_dim, projection_dim, num_multi_heads,
                 num_transformer_block, ffn_units, dropout, num_model_layer, classifier_units, include_top=False, pooling_mode="1D"):

        self.patch_size = patch_size
        self.data_dim = data_dim  # num_patches* projection_dim
        self.lattent_dim = lattent_dim
        self.projection_dim = projection_dim
        self.num_multi_heads = num_multi_heads
        self.num_transformer_block = num_transformer_block
        self.ffn_units = ffn_units
        self.dropout = dropout
        self.num_model_layer = num_model_layer
        self.classifier_units = classifier_units
        self.include_top = include_top
        self.pooling_mode = pooling_mode
        super(perceiver_architecture_V1, self).__init__(
            name="Perceiver_Architecture")

    # def build(self, input_shape):

        # create lattent array with init random values
        self.latent_array = self.add_weight(shape=(self.lattent_dim, self.projection_dim),
                                            initializer="random_normal", trainable=True)

        # create patching module
        self.num_patches = patches(self.patch_size)

        # create patch embedding encoded (position, content information) data input (K, V)
        self.patches_embedding = patch_content_position_encoding(
            self.data_dim, self.projection_dim)

        # Create cross-attention module
        self.cross_attention = cross_attention_module(self.lattent_dim, self.data_dim, self.projection_dim,
                                                      self.ffn_units, self.dropout)

        # Create Latten_transformer_Attention
        self.latent_transformer = latten_transformer_attention(self.lattent_dim, self.projection_dim, self.num_multi_heads,
                                                               self.num_transformer_block, self.ffn_units, self.dropout)

        if self.pooling_mode == "1D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        elif self.pooling_mode == "2D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        else:
            raise Exception("you not implement available pooling mode")

        if self.include_top == True:
            self.classification_head = create_classification_ffn(
                units_neuron=self.classifier_units, dropout_rate=self.dropout)

        #super(perceiver_architecture, self).build(input_shape)

    def call(self, inputs):
        # Augmentation option --> self-supervised processing outside
        # augmented= data_augmentation(inputs)
        # this for training using tf.data.Dataset
        # inputs, _ = inputs
        # create patches
        num_patches = self.num_patches(inputs)

        # embedding patches position content information
        embedding_patches = self.patches_embedding(num_patches)

        # passing input to cross attention
        cross_attention_input = {"latent_array": tf.expand_dims(self.latent_array, 0),
                                 "data_array": embedding_patches,
                                 }
        # Apply cross attention --> latent transform --> Stack multiple build deeper model
        for _ in range(self.num_model_layer):
            # Applying cross attention to INPUT
            latent_array = self.cross_attention(cross_attention_input)
            # apply latent attention to cross attention OUTPUT
            latent_array = self.latent_transformer(latent_array)
            # set the latent array out output to the next block
            cross_attention_input["latent_array"] = latent_array

        # Applying Global Average_pooling to generate [Batch_size, projection_dim] representation
        representation = self.global_average_pooling(latent_array)

        if self.include_top == True:
            representation = self.classification_head(representation)

        return representation


class perceiver_architecture_V2(tf.keras.Model):

    def __init__(self, patch_size, data_dim, lattent_dim, projection_dim, num_multi_heads,
                 num_transformer_block, ffn_units, dropout, num_model_layer, pooling_mode="1D"):

        self.patch_size = patch_size
        self.data_dim = data_dim  # num_patches* projection_dim
        self.lattent_dim = lattent_dim
        self.projection_dim = projection_dim
        self.num_multi_heads = num_multi_heads
        self.num_transformer_block = num_transformer_block
        self.ffn_units = ffn_units
        self.dropout = dropout
        self.num_model_layer = num_model_layer
        self.pooling_mode = pooling_mode
        super(perceiver_architecture_V2, self).__init__(
            name="Perceiver_Architecture")

    # def build(self, input_shape):

        # create lattent array with init random values
        self.latent_array = self.add_weight(shape=(self.lattent_dim, self.projection_dim),
                                            initializer="random_normal", trainable=True)

        # create patching module
        self.num_patches = patches(self.patch_size)

        # create patch embedding encoded (position, content information) data input (K, V)
        self.patches_embedding = patch_content_position_encoding(
            self.data_dim, self.projection_dim)

        # Create cross-attention module
        self.cross_attention = cross_attention_module(self.lattent_dim, self.data_dim, self.projection_dim,
                                                      self.ffn_units, self.dropout)

        # Create Latten_transformer_Attention
        self.latent_transformer = latten_transformer_attention(self.lattent_dim, self.projection_dim, self.num_multi_heads,
                                                               self.num_transformer_block, self.ffn_units, self.dropout)

        # if self.pooling_mode == "1D":
        #     self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        # elif self.pooling_mode == "2D":
        #     self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        # else:
        #     raise Exception("you not implement available pooling mode")

        # super(perceiver_architecture, self).build(input_shape)

    def call(self, inputs):
        # Augmentation option --> self-supervised processing outside
        #augmented= data_augmentation(inputs)
        # this for training using tf.data.Dataset
        #inputs, _ = inputs
        # create patches
        num_patches = self.num_patches(inputs)

        # embedding patches position content information
        embedding_patches = self.patches_embedding(num_patches)

        # passing input to cross attention
        cross_attention_input = {"latent_array": tf.expand_dims(self.latent_array, 0),
                                 "data_array": embedding_patches,
                                 }
        # Apply cross attention --> latent transform --> Stack multiple build deeper model
        for _ in range(self.num_model_layer):
            # Applying cross attention to INPUT
            latent_array = self.cross_attention(cross_attention_input)
            # apply latent attention to cross attention OUTPUT
            latent_array = self.latent_transformer(latent_array)
            # set the latent array out output to the next block
            cross_attention_input["latent_array"] = latent_array

        # Applying Global Average_pooling to generate [Batch_size, projection_dim] representation
            # Create a [batch_size, projection_dim] tensor.
        representation = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(latent_array)
        representation = tf.keras.layers.Flatten()(representation)
        #representation = self.global_average_pooling(representation)

        return representation


class convnet_perceiver_architecture(tf.keras.Model):

    def __init__(self,
                 # Conv_unroll paches_image
                 image_size, num_conv_layers, conv_position_embedding, spatial2projection_dim,
                 # Cross attention Module
                 data_dim, lattent_dim, projection_dim,
                 # For the Latten transformer and Model depth
                 num_multi_heads, num_transformer_block, num_model_layer,
                 # For model MLP (Pointwise Linear feed forward model)
                 ffn_units, dropout,  classifier_units,

                 # Configure the Rep output
                 include_top=False, pooling_mode="1D",):

        self.image_size = image_size
        self.conv_position_embedding = conv_position_embedding
        self.num_conv_layer = num_conv_layers
        self.spatial2projection_dim = spatial2projection_dim

        # Configure data
        self.data_dim_ = data_dim  # num_patches
        self.lattent_dim = lattent_dim
        self.projection_dim = projection_dim
        self.num_multi_heads = num_multi_heads
        self.num_transformer_block = num_transformer_block
        # Configure output
        self.ffn_units = ffn_units
        self.dropout = dropout
        self.num_model_layer = num_model_layer
        self.classifier_units = classifier_units
        self.include_top = include_top
        self.pooling_mode = pooling_mode
        super(convnet_perceiver_architecture, self).__init__(
            name="C_Conv_Perceiver_Architecture")

        # preprocessing for embedding position
        if self.conv_position_embedding:
            self.input_img_position_encode = tf.ones(
                (1, self.image_size, self.image_size, 3))

    def build(self, input_shape):
        # create lattent array with init random values
        self.latent_array = self.add_weight(shape=(self.lattent_dim, self.projection_dim),
                                            initializer="random_normal", trainable=True)

        ''' The modification is here'''
        # create patches from Conv
        self.num_patches = conv_unroll_patches_position_encoded(
            self.num_conv_layer, self.spatial2projection_dim)

        self.patches_position_encoding, self.data_dim = self.num_patches.conv_content_position_encoding(
            self.image_size)

        # create patching module
        # self.num_patches = patches(self.patch_size)
        # # create patch embedding encoded (position, content information) data input (K, V)
        # self.patches_embedding = patch_content_position_encoding(
        #     self.data_dim, self.projection_dim)

        # Create cross-attention module
        self.cross_attention = cross_attention_module(self.lattent_dim, self.data_dim, self.projection_dim,
                                                      self.ffn_units, self.dropout)

        # Create Latten_transformer_Attention
        self.latent_transformer = latten_transformer_attention(self.lattent_dim, self.projection_dim, self.num_multi_heads,
                                                               self.num_transformer_block, self.ffn_units, self.dropout)

        if self.pooling_mode == "1D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        elif self.pooling_mode == "2D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        else:
            raise Exception("you not implement available pooling mode")

        if self.include_top == True:
            self.classification_head = create_classification_ffn(
                units_neuron=self.classifier_units, dropout_rate=self.dropout)

        super(convnet_perceiver_architecture, self).build(input_shape)

    def call(self, inputs):
        # Augmentation option --> self-supervised processing outside
        # augmented= data_augmentation(inputs)
        # this for training using tf.data.Dataset
        # inputs, _ = inputs
        # create patches
        num_patches = self.num_patches(inputs)

        # embedding patches position content information
        linear_position_patches = self.patches_position_encoding
        patches_postions_encoded = tf.math.add(
            num_patches, linear_position_patches)
        print("this is data output shape", patches_postions_encoded.shape)

        # passing input to cross attention
        cross_attention_input = {"latent_array": tf.expand_dims(self.latent_array, 0),
                                 "data_array": patches_postions_encoded,
                                 }
        # Apply cross attention --> latent transform --> Stack multiple build deeper model
        for _ in range(self.num_model_layer):
            # Applying cross attention to INPUT
            latent_array = self.cross_attention(cross_attention_input)
            # apply latent attention to cross attention OUTPUT
            latent_array = self.latent_transformer(latent_array)
            # set the latent array out output to the next block
            cross_attention_input["latent_array"] = latent_array

        # Applying Global Average_pooling to generate [Batch_size, projection_dim] representation
        representation = self.global_average_pooling(latent_array)

        if self.include_top == True:
            representation = self.classification_head(representation)

        return representation


####################################################################################
'''2. Vision Transformer Architecture

        1. Patches Unroll can also re-use from Previous Perciever architecture
        2. Emebdding patches should incldue the
        3. Some optimal configure for building MLP 
        4. Building Projection Dimentions

    3. GOAL Further Design Axial Attention for ViT model 
'''
####################################################################################

# Using keras API


class self_attention_VIT(tf.keras.Model):
    '''Args
    patches_nums  will be equal to the number data-dimentions
    '''

    def __init__(self, patch_size, num_patches, projection_dim, transformer_layer,
                 number_attention_head, transformer_FPN_units, fft_units, dropout, num_classes, include_top=False, flatten_mode="flatten"):
        super(self_attention_VIT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.transformer_layer = transformer_layer
        self.number_attention_head = number_attention_head
        self.transformer_FPN_units = transformer_FPN_units
        self.fft_units = fft_units
        self.include_top = include_top
        self.dropout = dropout
        self.flatten_mode = flatten_mode
        self.num_classes = num_classes

    def build(self, input_shape):
        self.patches_unroll = patches(self.patch_size)
        self.patches_encoded = patch_content_position_encoding(
            self.num_patches, self.projection_dim)
        # self.global_average_pooling= tf.keras.layers.GlobalAveragePooling1D()
        # self.fallen_layer= tf.keras.layers.Flatten()
        super(self_attention_VIT, self).build(input_shape)

    def call(self, input):
        patches_unroll = self.patches_unroll(input)
        patches_encoded = self.patches_encoded(patches_unroll)

        for _ in range(self.transformer_layer):
            # Applying Layer Normalization
            x = tf.keras.layers.LayerNormalization(
                epsilon=1e-6)(patches_encoded)
            # Applying the multi-head Attention Layers
            x = tf.keras.layers.MultiHeadAttention(
                num_heads=self.number_attention_head, key_dim=self.projection_dim, dropout=0.1)(x, x)
            # Adding the skip connection
            x = tf.keras.layers.Add()([x, patches_encoded])
            # Normalization again
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-4)(x1)
            # passing Poitwise feed forward network
            x1 = create_ffn(units_neuron=self.transformer_FPN_units,
                            dropout_rate=self.dropout)(x1)
            patches_encoded = tf.keras.layers.Add()([x1, x])

        # Create [Batch_size, projection_dim] Tensor
        represenation = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)(patches_encoded)
        if self.flatten_mode == "flatten":
            represenation = tf.keras.layers.Flatten()(represenation)
        elif self.flatten_mode == "avpool1D":
            represenation = tf.keras.layers.GlobalAveragePooling1D()(represenation)
        # Further go through FFN processing
        feature = create_ffn(units_neuron=self.fft_units,
                             dropout_rate=self.dropout)
        if self.include_top:
            feature = tf.keras.layers.Dense(self.num_classes)(feature)
        return feature


####################################################################################
'''2. Vision Transformer Compact and Lite Version Implementation Architecture'''
####################################################################################


class compact_lite_vit(tf.keras.Model):
    pass


####################################################################################
'''3.Compact Convolution Transformer Architecture'''
####################################################################################
# Transformer not well inductive_bias-- let Conv helps

'''
1. Building the pathes extract using Conv-- Embedding positional encode
    # at embedding position --> can linear, learnable--Sinusoid, 
    # Axial positon encoding {Future improvemnt}

2. Building the Stochastic Depth regularization -- randomly drops a set of layers.. 
    # stochastice depth is a regularization technique is drop layer while dropout is drop Neurons inside each layers
    
3. FFN- Feed forward and MLP classifiy 
    # we can optional building MLP Mixer

'''

# Patches -- tokenized the images -- Using Conv instead patches VIT
# The same concept Unroll the Image with Conv + Flatten layer at the end


class ccttokenized(tf.keras.layers.Layer):
    #num_output_channels=[64, 128],
    def __init__(self, num_conv_layers, spatial2project_dim, positional_emb,
                 kernel_size=3, stride=1, padding=1, pooling_kernel_size=3, pooling_stride=2,
                 **kwargs):

        super(ccttokenized, self).__init__(**kwargs)
        # self.num_conv_layers = num_conv_layers
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # self.pooling_kernel_size = pooling_kernel_size
        # self.pooling_stride = pooling_stride

        # Conv 3x3 kernel stride over +pooling --> flattenned --> sequency of patches
        self.conv_model = tf.keras.Sequential()

        for i in range(num_conv_layers):

            self.conv_model.add(tf.keras.layers.Conv2D(
                spatial2project_dim[i], kernel_size, stride, padding="valid",
                use_bias=False, activation='relu', kernel_initializer='he_normal'))

        self.conv_model.add(tf.keras.layers.ZeroPadding2D(padding))
        self.conv_model.add(tf.keras.layers.MaxPool2D(
            pooling_kernel_size, pooling_stride, "same"))

        # Position Embedding optional I believe it will much support further when provide much information
        self.positional_emb = positional_emb

    def call(self, images):
        # Unroll images by conv_model
        outputs = self.conv_model(images)
        # the spatial dimension --> Flattened--> sequences.
        flatten_sequences = tf.reshape(
            outputs, (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]))

        print(
            f" Spatial_featuremap_2_projection_dim---> Flattend -->Unroll sequences: {flatten_sequences} ")
        return flatten_sequences

    # Position embedded should go along with the Conv kernel sequence
    # THis improve shoudl is my future Contribution
    @tf.function
    def linear_position_embedding(self, image_size):
        '''Args

        taken image_size inputs
        return linear_postion_encoded, sequence_len, 

        '''
        if self.positional_emb:
            print("Implement Linear Position Embedding")
            position_encode_image = tf.ones((1, image_size, image_size, 3))
            # using conv_model extract random position[sequ position change base random filter_val conv2D]
            sequence_posit = self.call(position_encode_image)
            print('this is Conv extract Flatten Sequences',
                  (sequence_posit.shape))
            sequence_pos_len = tf.shape(
                sequence_posit, out_type=tf.dtypes.int32)[1]
            print('this is paches_len', sequence_pos_len)
            # this will equal to number filter extract
            project_dim = tf.shape(
                sequence_posit, out_type=tf.dtypes.int32)[-1]
            print('this  is projection dim', project_dim)
            # ALl the image will have the same position --> THE SAME or Differ
            embed_linear_postion = tf.keras.layers.Embedding(
                input_dim=sequence_pos_len, output_dim=project_dim)

            print(
                f"Implemetn_Linear_Position encoding---- position_sequence_len: {sequence_pos_len}")

            return embed_linear_postion, sequence_pos_len

        else:
            return None

    def axial_position_embedding(self, image_size):
        raise NotImplementedError

# Stochastic Depth-- Similar Dropout Concpet


class stochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_layer, ):
        super(stochasticDepth, self).__init__()
        self.drop_layer = drop_layer

    def call(self, x, training=None):
        #print("this is tensor shape", x.shape)
        if training:
            keep_layer = 1 - self.drop_layer
            # shape = (tf.shape(x)[0], ) + (1, ) * \
            #     (len(tf.shape(x, out_type=tf.dtypes.int32))-1)
            shape = (tf.shape(x)[0], ) + (1, )*(len(x.shape)-1)
            #print("this is the tensor shape", shape)
            random_tensor = keep_layer + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_layer) * random_tensor

        return x

    def get_config(self):

        config = super(stochasticDepth, self).get_config().copy()
        config.update({
            'drop_layer': self.drop_layer,
        })

        return config


# 3. Define FFN- MLP
# Optional Using
# Actually Not Implement in the code
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

# API to build the model


class conv_transform(tf.keras.Model):
    '''args
    Noted the projection_dim= spatial2project_dim[-1]

    '''

    def __init__(self, num_class, image_size, num_conv_layers, spatial2project_dim, embedding_option,
                 transformer_blocks,  num_head_attention, projection_dim, ffn_units, stochastic_depth_rate, dropout, include_top):
        super(conv_transform, self).__init__()
        self.num_class = num_class
        self.include_top = include_top
        self.image_size = image_size
        self.num_head_attention = num_head_attention
        self.transformer_blocks = transformer_blocks
        self.projection_dim = projection_dim
        self.embedding_option = embedding_option
        self.ffn_units = ffn_units
        self.dropout_rate = dropout
        self.num_conv_layers = num_conv_layers
        self.spatial2project_dim = spatial2project_dim

        # unroll image to patches ++ Position encoding option
        self.patches = conv_unroll_patches_position_encoded(
            num_conv_layers, spatial2project_dim)
        # Using Cls layer seem the output only the object can't work we need output directly tensor
        # self.patches_position_encoded = conv_content_position_encoding_cls(image_size,
        #                                                                    num_conv_layers, spatial2project_dim)

        self.patches_position_encoded = self.patches.conv_content_position_encoding(
            image_size)

        # calculate S
        self.dpr = [x for x in np.linspace(
            0, stochastic_depth_rate, transformer_blocks)]

    def call(self, inputs):

        patches = self.patches(inputs)

        if self.embedding_option:
            # dummy_img_posit = tf.ones(
            #     (1, self.image_size, self.image_size, 3))
            # sequences_flatten = self.patches(dummy_img_posit)
            # sequences_flatten_out = tf.shape(sequences_flatten)[1]
            # projection_dim = tf.shape(sequences_flatten)[-1]

            # position_encoding = tf.keras.layers.Embedding(input_dim=sequences_flatten_out,
            #                                               output_dim=projection_dim)

            # positions = tf.range(start=0, limit=sequences_flatten_out, delta=1)
            # position_encoding_out = position_encoding(positions)
            position_encoding_out = self.patches_position_encoded
            print("This is encoding position output", position_encoding_out)

            patches = tf.add(patches, position_encoding_out)

        for i in range(self.transformer_blocks):
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(patches)
            # multi-head attention layers
            multi_head_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_head_attention, key_dim=self.projection_dim, dropout=0.1)(x1, x1)
            attention_output = stochasticDepth(
                self.dpr[i])(multi_head_attention)

            # adding skip connection
            x2 = tf.keras.layers.Add()([attention_output, patches])

            # Normalization
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x2)

            # MLP layers
            x4 = create_ffn(units_neuron=self.ffn_units,
                            dropout_rate=self.dropout_rate)
            x4 = x4(x3)

            # adding second skip connection
            x5 = stochasticDepth(self.dpr[i])(x4)

            patches = tf.keras.layers.Add()([x5, x2])

        # Original model with Head classify Model Layers
        representation = tf.keras.layers.LayerNormalization(
            epsilon=1e-5)(patches)

        attention_weights = tf.nn.softmax(
            tf.keras.layers.Dense(1)(representation), axis=1)
        weighted_represenation = tf.matmul(
            attention_weights, representation, transpose_a=True)
        weighted_represenation = tf.squeeze(weighted_represenation, -2)
        if self.include_top:
            # clasify output
            logits = tf.keras.layers.Dense(
                self.num_class)(weighted_represenation)

            return logits

        else:
            return weighted_represenation

# Using tf.keras.Model building the model


def conv_transform_v1(input_shape, num_class, image_size, num_conv_layers, spatial2project_dim, embedding_option,
                      transformer_blocks,  num_head_attention, projection_dim, ffn_units, stochastic_depth_rate, dropout, include_top):
    input = tf.keras.layers.Input(input_shape)

    # Conv patches unroll
    patches_sequence = conv_unroll_patches_position_encoded(
        num_conv_layers, spatial2project_dim)
    patches_sequence_out = patches_sequence(input)

    if embedding_option:
        embedded_position, _ = patches_sequence.conv_content_position_encoding(
            image_size)
        patches_sequence_out = tf.math.add(
            patches_sequence_out, embedded_position)
    # calculate the stochastic Depth probability
    dpr = [x for x in np.linspace(
        0, stochastic_depth_rate, transformer_blocks)]

    # Stack multiple blocks transformer
    for i in range(transformer_blocks):

        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-5)(patches_sequence_out)
        # Creat the attention head layer
        attention_out = tf.keras.layers.MultiHeadAttention(
            num_heads=num_head_attention, key_dim=projection_dim, dropout=0.1)(x, x)
        # adding skip connection
        attention_out = stochasticDepth(dpr[i])(attention_out)
        x2 = tf.keras.layers.Add()([attention_out, patches_sequence_out])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x2)
        # adding Poitwise Feed forward net
        x4 = create_ffn(units_neuron=ffn_units, dropout_rate=dropout)
        x4 = x4(x3)
        x4 = stochasticDepth(dpr[i])(x4)
        # Skip connection 2.
        patches_sequence_out = tf.keras.layers.Add()([x4, x2])

    # Original model with Head classify Model Layers
    representation = tf.keras.layers.LayerNormalization(
        epsilon=1e-5)(patches_sequence_out)

    attention_weights = tf.nn.softmax(
        tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_represenation = tf.matmul(
        attention_weights, representation, transpose_a=True)
    weighted_represenation = tf.squeeze(weighted_represenation, -2)
    if include_top:
        print('using top')
        # clasify output
        representation_out = tf.keras.layers.Dense(
            num_class)(weighted_represenation)

    else:
        representation_out = weighted_represenation

    model = tf.keras.Model(inputs=input, outputs=representation_out)
    return model
