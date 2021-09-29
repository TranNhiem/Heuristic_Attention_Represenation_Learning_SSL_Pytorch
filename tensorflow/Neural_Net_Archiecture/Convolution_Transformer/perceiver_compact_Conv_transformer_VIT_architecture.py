'''
Three Seprate Architecture of Conv-Transformer
1. Conv- Self-Attention ViT architecture 
2. Conv- Cross- Attention Architecture
3. CIT -- Cross-Covariance Attention (XCA) (Basicly Conv1 architecture)
https://arxiv.org/abs/2106.09681

### General Architecture Building Steps
1/--> Unroll image to small patches 
    + Patches unroll image 
    + Using Conv unroll image

2/ --> Position Embeddeding for attention mechanism --> 
    + (Conv-ViT -- Position embedding seem not effect too much)
    + Building Sinoudsoid position embedding 
    + linear position embedding 
    + other techniques for position embedding 

3/ --> Attention (mechanism)(cross attention -- self attention layer)
    
    + Model Depth ()
    + Model Width ()
    + Depth and width with scaling factors 

4/ --> Feature Embedding ouput --> Maxpooling -- Encoding flattent 

'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LayerNormalization

####################################################################################
'''Extract Patches Unroll the Image'''
####################################################################################

# Patches Extract


class patches(tf.keras.layers.Layer):
    '''
    args: Patch_size the size of crop you expect to Unroll image into sequences (size * size)

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
        flatten_sequences = tf.cast(flatten_sequences, tf.float32)

        return flatten_sequences

# Display function show the image patches Extraction


def display_pathches(image, IMG_SIZE, patch_size, unroll_method="tf_patches"):
    '''
    Args: 
        image: input image
        IMG_SIZE: 
        Patch_size: 
        Unroll_method: 
    Return 
        Sequence of patches 
    '''
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis("off")

    # Resize image and Convert to Tensor
    image_resize = tf.image.resize(
        tf.convert_to_tensor([image]), size=(IMG_SIZE, IMG_SIZE))
    if unroll_method == "tf_patches":
        # Unroll image to many patches

        patches_unroll = patches(patch_size, )(image_resize)
        # Information of Unroll IMAGE Corresponding with Patch
        print(f'IMG_SIZE: {IMG_SIZE} x {IMG_SIZE}')
        print(f'Implement Patch_size: {patch_size} x {patch_size}')
        print(f'Patches per image: {patches_unroll.shape[1]}')
        print(f'Elements per Patch: {patches_unroll.shape[-1]}')
        # number of rows and columns
        n = int(np.sqrt(patches.shape[1]))
    elif unroll_method == "convolution":
        '''
        Develope Visualization of Patch Unroll Here

        '''
        pass

    else:
        raise ValueError("Unroll method not in current support methods")

    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches_unroll[0]):
        axis = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")


####################################################################################
'''Patches Position Encoding '''
####################################################################################
# 1 tf.keras.layer.embedding Position Encoding
# LINEAR transform into Vector (Learnable Position encoding)

# 2 Learnable and Not Learnable Fourier position Encoding
# Multi Freq transforms (Perciever-Perceiver IO)

# 3. Position Sensitive Encoding (Axial DeepLap paper)

# 4. Dinstangle Position Encoding (DeBERTA paper)

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

# This Current implementation For Unroll and position Encoding


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
        flatten_sequences = tf.cast(flatten_sequences, tf.float32)
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


#####################################################################################
'''Create Feed Forward Network -- Dropout - Stochastic Drop'''
######################################################################################
# Feed forward network contain in Attention all you need 2 linear and 1 ReLu in middel
# Feed forwared Network for Transformer


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

# Stochastic Depth
# -- Similar Dropout Concept (Instead of dropout Neurons --> This dropout layers)


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


####################################################################################
'''-----Building Transformer Attention Type Module----'''
# 1. Convention Self-attention module for encoder --- Decoder

# 2. Cross-Attention Module (Perceiver/PerceiverIO)

# [3, 4] Under Development
# 3. Axial Attention Module (Axial DeepLAB paper)

# 4. Distangle Attention Module (DeBERTA paper)
####################################################################################

# 1 Conventional Self-Attention Module
# Attention Consideration the lattent_dim and ffn_units last later(Same or Differen??)


def latten_transformer_attention(lattent_dim, projection_dim, num_multi_head,
                                 num_transformer_block, ffn_units, dropout, stochastic_depth=False, dpr=None):
    '''
    Args:
        Lattent_dim: Latten Dimension is output from "Cross attention module"
        num_multi_heads: number multi-head attention for handle multiple part inputs --> Concatenate at the end
        num_transformer_block:  Stack multi-attention heads module multiple time on top each other 
        ffn_units: MLP model procesing output from attention module (list Units for multi layer - single number for 1 layer)

        dropout: dropout rate neuron unit of MLP model

    return
        Attention Encoder model -> output of self-attention model (Size output == Size Cross Attention Input)

    '''

    # Input of transformer (N)--> quadratic O(N*N)
    # Input shape=[1, latent_dim, projection_dim]
    inputs = tf.keras.layers.Input(shape=(lattent_dim, projection_dim))
    x0 = inputs
    # stack multiple attention encoder block
    for i_ in range(num_transformer_block):
        #x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x0)
        x = LayerNormalization(epsilon=1e-6)(x0)
        # create multi-head attention
        # attention_axes=None attention over all axes
        multi_head_out = tf.keras.layers.MultiHeadAttention(num_heads=num_multi_head,
                                                            key_dim=projection_dim, dropout=dropout)(x, x)

        if stochastic_depth:
            multi_head_out = stochasticDepth(dpr[i_])(multi_head_out)
        # adding skip connection (Between multi head previous layernorm --> Norm again)

        x1 = tf.keras.layers.Add()([multi_head_out, x])
        #x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x1)
        x2 = LayerNormalization(epsilon=1e-6)(x1)
        # Apply Feed forward network
        x3 = create_ffn(units_neuron=ffn_units, dropout_rate=dropout)
        if stochastic_depth:
            x3 = stochasticDepth(dpr[i_])(x3)
        x3 = x3(x2)

        # Adding skip connection (betwen Layersnorm and ffn_output)
        x0 = tf.keras.layers.Add()([x3, x2])

    # create stack block model
    model = tf.keras.Model(inputs=inputs, outputs=x0)

    return model

# 2 Cross-Attention Module


def cross_attention_module(lattent_dim, data_dim, projection_dim, ffn_units, dropout):
    '''
    Args:
        latten_dim: Reduce dimension you expected to
        data_dim: Length unroll the image (num_patchets) 1xD sequence << (Original paper Using Width* High)
        ffn_units: MLP model layers len(ffn_units), # neuron= value of element inside single integer 1 layer
        dropout: percentages neuron dropout in the last MLP layer

    Return 
        the output is metrix (M*N) N is the latten Dimension M is data_dim input Dimension
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
    # lattent_array = tf.keras.layers.LayerNormalization(
    #     epsilon=1e-6)(inputs["latent_array"])
    lattent_array = LayerNormalization(
        epsilon=1e-6)(inputs["latent_array"])

    # data_array = tf.keras.layers.LayerNormalization(
    #     epsilon=1e-6)(inputs["data_array"])
    data_array = LayerNormalization(
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
    # attention_output_norm = tf.keras.layers.LayerNormalization(
    #     epsilon=1e-6)(attention_output)
    attention_output_norm = LayerNormalization(
        epsilon=1e-6)(attention_output)
    # Apply Feedforward Network
    ffn = create_ffn(units_neuron=ffn_units, dropout_rate=dropout)
    ffn = ffn(attention_output_norm)

    # adding skip connection 2: between attention output and FFN model
    outputs = tf.keras.layers.Add()([ffn, attention_output_norm])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


####################################################################################
'''1. Building Conv-Transformer Architecture

##  The important of Covolution Transformer
 
+1. The Stochastic Dropout (Regularization for self-Attention Blocks)

+2. Patching Unroll the image by Convolution (Multi-Layer Convolution to unroll)
        (1-2 Conv2D 3x3 size filter seem to work best)

+3. The output Representation Vector then using Sequence Pooling --> Not Maxpooling or Average Pooling
(Check out the performance and efficient Implement these two methods)

'''
####################################################################################
# 1. The cross-attention epxect a (lattent_dim, projection_dim) --> latten array
# 2. data array (data_dim, proction_dim) --> data arrray
# Original paper data array= Width* Height -- Unroll patches data_dim = 1*D sequences of patches

# ==> Dotproduct (latten_array, data_array) --> (latten_dim, projection_dim)
# Q Generated from lattent array, K, V generated from the encoded image
# data_dim will equal to number of patches after unrol the image.
# 3. Adding the SequencePooling and StochasticDepth for some addition improvemnt


class convnet_perceiver_architecture(tf.keras.Model):

    def __init__(self,
                 # Conv_unroll paches_image
                 IMG_SIZE, num_conv_layers, conv_position_embedding, spatial2projection_dim,
                 # Cross attention Module
                 lattent_dim, projection_dim,
                 # For the Latten transformer and Model depth
                 num_multi_heads, num_transformer_block, num_model_layer,
                 # For model MLP (Pointwise Linear feed forward model)
                 ffn_units, dropout,  classifier_units,

                 # Configure the Rep output, addittion stochastic dropout

                 include_top=False, pooling_mode="1D", stochastic_depth=False, stochastic_depth_rate=0.1

                 ):

        super(convnet_perceiver_architecture, self).__init__(
            name="Conv_Perceiver_Arch")

        self.IMG_SIZE = IMG_SIZE
        self.conv_position_embedding = conv_position_embedding
        self.num_conv_layer = num_conv_layers
        self.spatial2projection_dim = spatial2projection_dim

        # Configure data
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

        # preprocessing for embedding position
        # if self.conv_position_embedding:
        #     self.input_img_position_encode = tf.ones(
        #         (1, self.IMG_SIZE, self.IMG_SIZE, 3))

        # Configure for Stochastic Depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.stochastic_depth = stochastic_depth
        self.dpr = None
        if stochastic_depth:
            # calculate Stochastic propability
            self.stochastic_depth = stochastic_depth
            self.dpr = [x for x in np.linspace(
                0, stochastic_depth_rate, num_transformer_block)]

    def build(self, input_shape):
        # create lattent array with init random values
        self.latent_array = self.add_weight(shape=(self.lattent_dim, self.projection_dim),
                                            initializer="random_normal", trainable=True)

        ''' The modification is here'''
        # create patches from Conv
        self.num_patches = conv_unroll_patches_position_encoded(
            self.num_conv_layer, self.spatial2projection_dim)

        self.patches_position_encoding, self.data_dim = self.num_patches.conv_content_position_encoding(
            self.IMG_SIZE)

        # create tf.image_patches module
        # self.num_patches = patches(self.patch_size)
        # # create patch embedding encoded (position, content information) data input (K, V)
        # self.patches_embedding = patch_content_position_encoding(
        #     self.data_dim, self.projection_dim)

        # Create cross-attention module
        self.cross_attention = cross_attention_module(self.lattent_dim, self.data_dim, self.projection_dim,
                                                      self.ffn_units, self.dropout,)

        # Create Latten_transformer_Attention
        self.latent_transformer = latten_transformer_attention(self.lattent_dim, self.projection_dim, self.num_multi_heads,
                                                               self.num_transformer_block, self.ffn_units, self.dropout, stochastic_depth=self.stochastic_depth, dpr=self.dpr)

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

        if self.conv_position_embedding:

            # embedding patches position content information learnable
            linear_position_patches = self.patches_position_encoding
            num_patches = tf.math.add(
                num_patches, linear_position_patches)

        print("Debug Covnet Unroll Patches Output",
              num_patches.shape)

        # passing input to cross attention
        cross_attention_input = {"latent_array": tf.expand_dims(self.latent_array, 0),
                                 "data_array": num_patches,
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

        if self.pooling_mode == "1D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
            representation = self.global_average_pooling(latent_array)
        # has to modify the output to use in 2D pooling
        # elif self.pooling_mode == "2D":
        #     self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        #     representation = self.global_average_pooling(latent_array)

        elif self.pooling_mode == "sequence_pooling":
            representation = tf.keras.layers.LayerNormalization(
                epsilon=1e-5)(latent_array)
            attention_weights = tf.nn.softmax(
                tf.keras.layers.Dense(1)(representation), axis=1)

            weighted_representation = tf.matmul(
                attention_weights, representation, transpose_a=True)
            representation = tf.squeeze(weighted_representation, -2)

        else:
            raise Exception("you're pooling mode not available")

        if self.include_top == True:
            representation = self.classification_head(representation)

        return representation


####################################################################################
'''2.Compact Convolution self-Attention Transformer Architecture'''
####################################################################################
# Transformer not well inductive_bias-- let Conv helps
'''
# Patches -- tokenized the images -- Using Conv instead patches VIT

# Building the Stochastic Depth regularization -- randomly drops a set of layers.. 
    # stochastice depth is a regularization technique is drop layer while dropout is drop Neurons inside each layers 
'''
####################################################################################


class conv_transform_VIT(tf.keras.Model):
    '''args
    Noted the projection_dim= spatial2project_dim[-1]

    '''

    def __init__(self, num_class, IMG_SIZE, num_conv_layers, spatial2project_dim, embedding_option, projection_dim,
                 num_transformer_blocks, num_head_attention, ffn_units, classification_unit,
                 dropout, stochastic_depth=False, stochastic_depth_rate=0.1,
                 include_top='False', pooling_mode="1D",
                 ):
        super(conv_transform_VIT, self).__init__(name="C_Conv_Perceiver_Arch")

        # For classification Configure
        self.num_class = num_class
        self.include_top = include_top
        self.pooling_mode = pooling_mode
        self.IMG_SIZE = IMG_SIZE

        # Attention module Configure
        self.num_head_attention = num_head_attention
        self.num_transformer_blocks = num_transformer_blocks
        self.projection_dim = projection_dim
        self.embedding_option = embedding_option
        # MPL configure
        self.ffn_units = ffn_units
        self.classifier_units = classification_unit
        self.dropout_rate = dropout

        # Convolution patches unroll Configure
        self.num_conv_layers = num_conv_layers
        self.spatial2project_dim = spatial2project_dim

        # Configure for Stochastic Depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dpr = None
        if stochastic_depth:
            # calculate Stochastic propability
            self.stochastic_depth = stochastic_depth
            self.dpr = [x for x in np.linspace(
                0, stochastic_depth_rate, num_transformer_blocks)]

    def build(self, input_shape):

        self.num_patches = conv_unroll_patches_position_encoded(
            self.num_conv_layers, self.spatial2project_dim)

        # embedding patches position content information learnable
        linear_position_patches = self.num_patches.conv_content_position_encoding(
            self.IMG_SIZE)

        self.patches_postions_encoded = tf.math.add(
            self.num_patches, linear_position_patches)

        print("this is data output shape", self.patches_postions_encoded.shape)

        # Classification Head Configure
        if self.include_top == True:
            self.classification_head = create_classification_ffn(
                units_neuron=self.classifier_units, dropout_rate=self.dropout)

        super(conv_transform_VIT, self).build(input_shape)

    def call(self, inputs):
        # Augmentation option --> self-supervised processing outside
        # create patches
        num_patches = self.num_patches(inputs)
        # embedding patches position content information learnable
        linear_position_patches = self.patches_postions_encoded
        patches_postions_encoded = tf.math.add(
            num_patches, linear_position_patches)

        patches_sequences = {"img_patches_seq": patches_postions_encoded, }

        # Create transformer_self-Attention
        latent_transformer = latten_transformer_attention(num_patches, self.projection_dim, self.num_head_attention,
                                                          self.num_transformer_blocks, self.ffn_units, self.dropout_rate, stochastic_depth=self.stochastic_depth, dpr=self.dpr)

        # Apply cross attention --> latent transform --> Stack multiple build deeper model
        for _ in range(self.num_transformer_blocks):
            # Applying cross attention to INPUT
            # apply latent attention to cross attention OUTPUT
            self_attention_out = latent_transformer(patches_sequences)
            # set the latent array out output to the next block
            patches_sequences["img_patches_seq"] = self_attention_out

        # Applying Global Average_pooling to generate [Batch_size, projection_dim] representation

        if self.pooling_mode == "1D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
            representation = self.global_average_pooling(self_attention_out)

        elif self.pooling_mode == "2D":
            self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
            representation = self.global_average_pooling(self_attention_out)

        elif self.pooling_mode == "sequence_pooling":
            representation = tf.keras.layers.LayerNormalization(
                epsilon=1e-5)(self_attention_out)
            attention_weights = tf.nn.softmax(
                tf.keras.layers.Dense(1)(representation), axis=1)

            weighted_representation = tf.matmul(
                attention_weights, representation, transpose_a=True)
            representation = tf.squeeze(weighted_representation, -2)

        else:
            raise Exception("you're pooling mode not available")

        if self.include_top == True:
            representation = self.classification_head(representation)

        return representation

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
