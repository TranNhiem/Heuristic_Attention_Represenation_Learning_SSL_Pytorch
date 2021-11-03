import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense,Flatten, Add, TimeDistributed, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D,MaxPooling1D,GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Conv3D,MaxPooling3D,GlobalAveragePooling3D, GlobalMaxPooling3D
from tensorflow.keras.layers import Layer


from GroupNorm import GroupNormalization

def define_NormLayers(norm):
    if norm=="BatchNorm":
        return  BatchNormalization
    elif norm=="GroupNorm":
        return GroupNormalization
    else:
        raise Exception("Normalization that you specify is invalid! Current value:",norm)


def define_ConvLayer(mode):
    if mode=="2D" or mode=="TimeD":
        return Conv2D
    elif mode=="1D":
        return Conv1D
    elif mode=="3D":
        return Conv3D
    else:
        raise Exception("Convolution mode that you specify is invalid! Current value:",mode)


def define_Pooling(mode):
    if mode=="2D" or mode=="TimeD":
        return MaxPooling2D
    elif mode=="1D":
        return MaxPooling1D
    elif mode=="3D":
        return MaxPooling3D
    else:
        raise Exception("Convolution mode that you specify is invalid! Current value:",mode)



def define_GlobalPooling(mode, pooling):
    if (mode=="2D" or mode=="TimeD") and pooling=="max":
        return GlobalMaxPooling2D
    elif mode=="1D"  and pooling=="max":
        return GlobalMaxPooling1D
    elif mode=="3D" and pooling=="max":
        return GlobalMaxPooling3D
    elif (mode=="2D" or mode=="TimeD") and pooling=="ave":
        return GlobalAveragePooling2D
    elif mode=="1D"  and pooling=="ave":
        return GlobalAveragePooling1D
    elif mode=="3D" and pooling=="ave":
        return GlobalAveragePooling3D    



class Conv_stage1_block(tf.keras.Model):
    def __init__(self, filters, strides=2, mode="2D", norm="BatchNorm",kernel_initializer='he_normal',name=None):  
        super(Conv_stage1_block,  self).__init__(name=name)
        NormLayer = define_NormLayers(norm) # Define Normalization Layers
        ConvLayer = define_ConvLayer(mode) #Define ConvLayer
        MaxPooling = define_Pooling(mode) # Define Pooling
        if mode=="1D" or mode=="2D" or mode=="3D":
            self.conv1 = ConvLayer(filters, kernel_size=7,strides=strides,kernel_initializer=kernel_initializer, padding='same')
            self.bn1 = NormLayer()
            self.act1 = Activation('relu')
            self.pool1 = MaxPooling(pool_size=3, strides=2,padding="same")
        elif mode=="TimeD":
            self.conv1 = TimeDistributed(ConvLayer(filters, kernel_size=7,kernel_initializer=kernel_initializer,strides=strides, padding='same'))
            self.bn1 = TimeDistributed(NormLayer())
            self.act1 = TimeDistributed(Activation('relu'))
            self.pool1 = TimeDistributed(MaxPooling(pool_size=(3,3), strides=(2,2),padding="same"))

    def call(self, x): 
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act1(h)
        output = self.pool1(h)
        return output


class Identity_bottleneck_block(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, mode="2D", norm="BatchNorm",kernel_initializer='he_normal' ,name=None):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        super(Identity_bottleneck_block,  self).__init__(name=name)
        NormLayer = define_NormLayers(norm) # Define Normalization Layers
        ConvLayer = define_ConvLayer(mode)
        filters1, filters2, filters3 = filters
        if mode=="1D" or mode=="2D" or mode=="3D":
            self.bn1 = NormLayer()
            self.relu1 = Activation('relu')
            self.conv1 = ConvLayer(filters1, 1, kernel_initializer=kernel_initializer,padding='same')
            self.bn2 = NormLayer()
            self.relu2 = Activation('relu')
            self.conv2 = ConvLayer(filters2,  kernel_size, kernel_initializer=kernel_initializer,padding='same')
            self.bn3 = NormLayer()
            self.relu3 = Activation('relu')
            self.conv3 = ConvLayer(filters3, 1, kernel_initializer=kernel_initializer,padding='same')
        elif mode=="TimeD":
            self.bn1 = TimeDistributed(NormLayer())
            self.relu1 = TimeDistributed(Activation('relu'))
            self.conv1 = TimeDistributed(ConvLayer(filters1, (1,1), kernel_initializer=kernel_initializer,padding='same'))
            self.bn2 = TimeDistributed(NormLayer())
            self.relu2 = TimeDistributed(Activation('relu'))
            self.conv2 = TimeDistributed(ConvLayer(filters2,  kernel_size, kernel_initializer=kernel_initializer,padding='same'))
            self.bn3 = TimeDistributed(NormLayer())
            self.relu3 = TimeDistributed(Activation('relu'))
            self.conv3 = TimeDistributed(ConvLayer(filters3, (1,1), kernel_initializer=kernel_initializer,padding='same'))

        self.add = Add()

    def call(self, x):
        residual = x
        h = self.bn1(x)
        h = self.relu1(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        h = self.bn3(h)
        h = self.relu3(h)
        h = self.conv3(h)
        # Merge
        output = self.add([residual, h])
        return output



class Conv_bottleneck_block(tf.keras.Model):
    def __init__(self,filters, kernel_size=3, strides=2, mode="2D",norm="BatchNorm",kernel_initializer='he_normal' , name=None):
        """A block that has a conv layer at shortcut.
        # Arguments
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        super(Conv_bottleneck_block,  self).__init__(name=name)
        NormLayer = define_NormLayers(norm) # Define Normalization Layers
        ConvLayer = define_ConvLayer(mode) # Define ConvLayer
        filters1, filters2, filters3 = filters
        if mode=="1D" or mode=="2D" or mode=="3D":
            # Left
            self.bn1 = NormLayer()
            self.relu1 = Activation('relu')
            self.conv1 = ConvLayer(filters1, 1, strides=strides,kernel_initializer=kernel_initializer,padding='same')
            self.bn2 = NormLayer()
            self.relu2 = Activation('relu')
            self.conv2 = ConvLayer(filters2,  kernel_size, kernel_initializer=kernel_initializer,padding='same')
            self.bn3 = NormLayer()
            self.relu3 = Activation('relu')
            self.conv3 = ConvLayer(filters3, 1, kernel_initializer=kernel_initializer,padding='same')
            #Right(shortcut)
            self.s_bn = NormLayer()
            self.s_conv = ConvLayer(filters3, 1, strides=strides,
                                            kernel_initializer=kernel_initializer,padding='same')
        elif mode == "TimeD":
            # Left
            self.bn1 = TimeDistributed(NormLayer())
            self.relu1 = TimeDistributed(Activation('relu'))
            self.conv1 = TimeDistributed(ConvLayer(filters1, (1,1), strides=strides,kernel_initializer=kernel_initializer,padding='same'))
            self.bn2 = TimeDistributed(NormLayer())
            self.relu2 = TimeDistributed(Activation('relu'))
            self.conv2 = TimeDistributed(ConvLayer(filters2,  kernel_size, kernel_initializer=kernel_initializer,padding='same'))
            self.bn3 = TimeDistributed(NormLayer())
            self.relu3 = TimeDistributed(Activation('relu'))
            self.conv3 = TimeDistributed(ConvLayer(filters3, (1,1), kernel_initializer=kernel_initializer,padding='same'))
            #Right(shortcut)
            self.s_bn = TimeDistributed(NormLayer())
            self.s_conv = TimeDistributed(ConvLayer(filters3, (1,1), strides=strides, kernel_initializer=kernel_initializer,padding='same'))
 
        self.add = Add()

    def call(self, x):
        residual = x
        #Left
        h = self.bn1(x)
        h = self.relu1(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        h = self.bn3(h)
        h = self.relu3(h)
        h = self.conv3(h)
        #Right
        residual = self.s_bn(residual)
        residual = self.s_conv(residual)
        # Merge
        output = self.add([residual, h])
        return output



class Identity_basic_block(tf.keras.Model):
    def __init__(self, filters,kernel_size=3,  mode="2D", norm="BatchNorm",kernel_initializer='he_normal' , name=None):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        super(Identity_basic_block,  self).__init__(name=name)
        NormLayer = define_NormLayers(norm) # Define Normalization Layers
        ConvLayer = define_ConvLayer(mode) # Define ConvLayer
        filters1, filters2 = filters
        if mode=="1D" or mode=="2D" or mode=="3D":
            self.bn1 = NormLayer()
            self.relu1 = Activation('relu')
            self.conv1 = ConvLayer(filters1, kernel_size, kernel_initializer=kernel_initializer,padding='same')
            self.bn2 = NormLayer()
            self.relu2 = Activation('relu')
            self.conv2 = ConvLayer(filters2,  kernel_size, kernel_initializer=kernel_initializer,padding='same')
        elif mode=="TimeD":
            self.bn1 = TimeDistributed(NormLayer())
            self.relu1 = TimeDistributed(Activation('relu'))
            self.conv1 = TimeDistributed(ConvLayer(filters1, kernel_size, kernel_initializer=kernel_initializer,padding='same'))
            self.bn2 = TimeDistributed(NormLayer())
            self.relu2 = TimeDistributed(Activation('relu'))
            self.conv2 = TimeDistributed(ConvLayer(filters2,  kernel_size, kernel_initializer=kernel_initializer,padding='same'))

        self.add = Add()

    def call(self, x):
        residual = x
        h = self.bn1(x)
        h = self.relu1(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        # Merge
        output = self.add([residual, h])
        return output




class Conv_basic_block(tf.keras.Model):
    def __init__(self,filters, kernel_size=3, strides=2, mode="2D", norm="BatchNorm",kernel_initializer='he_normal', name=None):        
        """A block that has a conv layer at shortcut.
        # Arguments
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        super(Conv_basic_block,  self).__init__(name=name)
        NormLayer = define_NormLayers(norm) # Define Normalization Layers
        ConvLayer = define_ConvLayer(mode)  # Define ConvLayer
        filters1, filters2 = filters
        if mode=="1D" or mode=="2D" or mode=="3D":
            # Left
            self.bn1 = NormLayer()
            self.relu1 = Activation('relu')
            self.conv1 = ConvLayer(filters1, 1, strides=strides,kernel_initializer=kernel_initializer,padding='same')
            self.bn2 = NormLayer()
            self.relu2 = Activation('relu')
            self.conv2 = ConvLayer(filters2,  kernel_size, kernel_initializer=kernel_initializer,padding='same')
            #Right(shortcut)
            self.s_bn = NormLayer()
            self.s_conv = ConvLayer(filters2, 1, strides=strides,kernel_initializer=kernel_initializer,padding='same')
        elif mode=="TimeD":
            # Left
            self.bn1 = TimeDistributed(NormLayer())
            self.relu1 = TimeDistributed(Activation('relu'))
            self.conv1 = TimeDistributed(ConvLayer(filters1, (1,1), strides=strides,kernel_initializer=kernel_initializer,padding='same'))
            self.bn2 = TimeDistributed(NormLayer())
            self.relu2 = TimeDistributed(Activation('relu'))
            self.conv2 = TimeDistributed(ConvLayer(filters2,  kernel_size, kernel_initializer=kernel_initializer,padding='same'))
            #Right(shortcut)
            self.s_bn = TimeDistributed(NormLayer())
            self.s_conv = TimeDistributed(ConvLayer(filters2, (1,1), strides=strides,kernel_initializer=kernel_initializer,padding='same'))
            
        self.add = Add()

    def call(self, x):
        #Left
        h = self.bn1(x)
        h = self.relu1(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        #Right
        residual = self.s_bn(x)
        residual = self.s_conv(residual)
        # Merge
        output = self.add([residual, h])
        return output



class Fin_layer(tf.keras.Model):
    def __init__(self,mode="2D", class_num=1000, include_top=True, pooling='avg', name=None):
        super(Fin_layer,  self).__init__(name=name)
        self.include_top = include_top
        self.mode=mode
        GlobalPooling = define_GlobalPooling(mode, pooling)
        if mode=="1D" or mode=="2D" or mode=="3D":
            #Pooling setting
            self.gp = GlobalPooling()
            if self.include_top:
                self.dense = Dense(class_num, 'softmax')         
        elif mode=="TimeD":
            self.gp = TimeDistributed(GlobalPooling())           
            if self.include_top:
                self.flat = Flatten()
                self.dense = Dense(class_num, 'softmax')            

    def call(self, x):
        output = self.gp(x)
        if self.include_top and (self.mode=="1D" or self.mode=="2D" or self.mode=="3D"):
            output = self.dense(output)
        if self.include_top and self.mode=="TimeD":
            output = self.flat(output)
            output = self.dense(output)
        return output



class BuildResnet(tf.keras.Model):
    def __init__(self, class_num=1000, include_top=True, pooling='ave', mode="2D", width_scale=1, norm="BatchNorm",kernel_initializer='he_normal', name=None):
        super(BuildResnet,  self).__init__(name=name)
        if not (mode=="1D" or mode=="2D" or mode=="TimeD" or mode=="3D"):
            raise Exception("'mode' value is invalid. you should use '1D' or '2D' or '3D' or 'TimeD'. Current value :",mode)
        if not (pooling=="ave" or pooling=="max" or pooling==None):
            raise Exception("'pooling' value is invalid. you should use 'ave' or 'max' or None. Current value :",pooling)       
        if not (include_top==True or include_top==False):
            raise Exception("'include_top' value is invalid. you should use bool value. Current value :",include_top)       
        self.pooling = pooling
        self.width_scale =width_scale
        filters_stage=[64, 128, 256, 512]

        self.stage_filter= [filter_*self.width_scale for filter_ in filters_stage] 
        print(f"your width_scale {  self.width_scale} & filter stage { self.stage_filter}")

        
        if name == "ResNet18":
            self.stage_filters =  self.stage_filter
            self.block_type = "basic"
            self.reptitions = [2, 2, 2, 2]
        elif name == "ResNet34":
            self.stage_filters =  self.stage_filter
            self.block_type = "basic"
            self.reptitions = [3, 4, 6, 3]
        elif name=="ResNet50":
            self.stage_filters =  self.stage_filter
            self.block_type = "bottleneck"
            self.reptitions = [3, 4, 6, 3]
        elif name=="ResNet101":
            self.stage_filters = self.stage_filter
            self.block_type = "bottleneck"
            self.reptitions = [3, 4, 23, 3]
        elif name=="ResNet152":
            self.stage_filters = self.stage_filter
            self.block_type = "bottleneck"
            self.reptitions = [3, 8, 36, 3]
        else:
            raise Exception(" Name Error! you can use ResNet18,ResNet34,ResNet50,ResNet101, or ResNet152. Current name:",name)

        if self.block_type=="basic":
            IdBlock = Identity_basic_block
            ConvBlock = Conv_basic_block
            all_filters = []
            for s_f in self.stage_filters:
                all_filters.append([s_f, s_f])

        elif self.block_type=="bottleneck":
            IdBlock = Identity_bottleneck_block
            ConvBlock = Conv_bottleneck_block
            all_filters = []
            for s_f in self.stage_filters:
                all_filters.append([s_f, s_f, s_f*4])


        # stage1 
        self.conv1 = Conv_stage1_block(filters=all_filters[0][0],mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        # stage2
        self.stage2_convs = {}
        self.stage2_convs[0] = ConvBlock(filters=all_filters[0],strides=1,mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        for rep in range(1,self.reptitions[0]):
            self.stage2_convs[rep] = IdBlock(filters=all_filters[0],mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        # stage3
        self.stage3_convs = {}
        self.stage3_convs[0] = ConvBlock(filters=all_filters[1],mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        for rep in range(1,self.reptitions[1]):
            self.stage3_convs[rep] = IdBlock(filters=all_filters[1],mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        # stage4
        self.stage4_convs = {}
        self.stage4_convs[0] = ConvBlock(filters=all_filters[2],mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        for rep in range(1,self.reptitions[2]):
            self.stage4_convs[rep] = IdBlock(filters=all_filters[2],mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        # stage5
        self.stage5_convs = {}
        self.stage5_convs[0] = ConvBlock(filters=all_filters[3],mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        for rep in range(1,self.reptitions[3]):
            self.stage5_convs[rep] = IdBlock(filters=all_filters[3],mode=mode,norm=norm,kernel_initializer=kernel_initializer)
        # Final Layer
        if self.pooling!=None:
            self.fin = Fin_layer(mode=mode, include_top=include_top, class_num=class_num, pooling=self.pooling)


    def call(self, x):
        # stage1
        h = self.conv1(x)
        # stage2
        for rep in range(self.reptitions[0]):
            h = self.stage2_convs[rep](h)
        # stage3
        for rep in range(self.reptitions[1]):
            h = self.stage3_convs[rep](h)
        # stage4
        for rep in range(self.reptitions[2]):
            h = self.stage4_convs[rep](h)
        # stage5
        for rep in range(self.reptitions[3]):
            h = self.stage5_convs[rep](h)
        # Final stage
        if self.pooling!=None:
            output = self.fin(h)
            return output
        else:
            return h