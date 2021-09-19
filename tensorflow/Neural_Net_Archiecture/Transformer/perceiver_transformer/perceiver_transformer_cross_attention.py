import tensorflow as tf

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

    ffn_layers.append(tf.keras.layers.Dense(units=units_neuron[-1]))
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
        print(patches.shape)
        return patches


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
        encoding = tf.math.add(self.projection(patches), self.position_encoding(
            positions), name="Encoding_patches")
        return encoding


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
            self.classification_head = create_ffn(
                units_neuron=self.classifier_units, dropout_rate=self.dropout)

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
        representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(latent_array)
        representation = tf.keras.layers.Flatten()(representation)
        #representation = self.global_average_pooling(representation)


        return representation

