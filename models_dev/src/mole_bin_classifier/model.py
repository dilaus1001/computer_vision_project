import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal

@tf.keras.utils.register_keras_serializable(package='HSwish act_func')
class HSwish(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = 'hswish'

    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0

@tf.keras.utils.register_keras_serializable(package='SE')
class SqueezeExcite(layers.Layer):
    def __init__(self, channels, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = max(1, channels // reduction)
        self._name = f'squeeze_excite_{channels}'

    def build(self, input_shape):
        self.reduce = layers.Dense(
            self.reduced_channels,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False,
            name=f'reduce_{self.name}'
        )
        
        self.expand = layers.Dense(
            self.channels,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False,
            name=f'expand_{self.name}'
        )

        super().build(input_shape)
    
    def call(self, inputs):
        squeezed = tf.reduce_mean(inputs, axis=[1, 2])
        
        reduced = self.reduce(squeezed)
        excited = self.expand(reduced)
        excited = tf.reshape(excited, [-1, 1, 1, self.channels])
        
        return inputs * excited
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction': self.reduction,
        })
        return config

@tf.keras.utils.register_keras_serializable(package='Block')
class Block(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 expansion_factor=1, se_ratio=0, activation='relu', **kwargs):
        super().__init__(**kwargs)
        
        # Store configuration parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion_factor = expansion_factor
        self.se_ratio = se_ratio
        self.activation_type = activation
        self.expanded_channels = in_channels * expansion_factor
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self._name = f'block_{in_channels}_{out_channels}_{stride}'
    
    def build(self, input_shape):
        # 1. Expansion phase
        if self.expansion_factor != 1:
            self.expand_conv = layers.Conv2D(
                self.expanded_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                name=f'expand_conv_{self.name}'
            )
            self.expand_bn = layers.BatchNormalization(name=f'expand_bn_{self.name}')
            self.expand_activation = (
                HSwish() if self.activation_type == 'hswish'
                else layers.ReLU(name=f'expand_relu_{self.name}')
            )
        
        # 2. Depthwise phase
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            use_bias=False,
            name=f'depthwise_conv_{self.name}'  
        )
        self.depthwise_bn = layers.BatchNormalization(name=f'depthwise_bn_{self.name}')
        self.depthwise_activation = (
            HSwish() if self.activation_type == 'hswish'
            else layers.ReLU(name=f'depthwise_relu_{self.name}')
        )
        
        # 3. Squeeze-and-Excite
        if self.se_ratio:
            self.se = SqueezeExcite(
                self.expanded_channels,
                reduction=4,
                name=f'squeeze_excite_{self.name}'
            )
        
        # 4. Projection phase
        self.project_conv = layers.Conv2D(
            self.out_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=f'project_conv_{self.name}'
        )
        self.project_bn = layers.BatchNormalization(name=f'project_bn_{self.name}')
        
        self.built = True
        super().build(input_shape)
    
    def call(self, inputs, training=None):
            x = inputs
            
            # Expansion phase
            if self.expansion_factor != 1:
                x = self.expand_conv(x)
                x = self.expand_bn(x, training=training)
                x = self.expand_activation(x)
            
            # Depthwise convolution
            x = self.depthwise_conv(x)
            x = self.depthwise_bn(x, training=training)
            x = self.depthwise_activation(x)
            
            # Squeeze-and-Excite
            if self.se_ratio:
                x = self.se(x)
            
            # Projection
            x = self.project_conv(x)
            x = self.project_bn(x, training=training)
            
            # Residual connection
            if self.use_residual:
                return x + inputs
            return x
    
    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expansion_factor': self.expansion_factor,
            'se_ratio': self.se_ratio,
            'activation': self.activation_type
        })
        return config
    

def create_mobile_model(input_shape=(224, 224, 3), num_classes=1):
    inputs = layers.Input(input_shape)

    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = HSwish()(x)
    
    # Architecture
    block_configs = [
        # in_ch, out_ch, kernel, stride, exp_factor, se_ratio, activation
        (16, 16, 3, 2, 1, 0, 'relu'),      # Initial spatial reduction
        (16, 24, 3, 2, 4, 0, 'relu'),      # Increase channels
        (24, 24, 3, 1, 4, 0, 'relu'),      # Process features
        (24, 40, 5, 2, 4, 1, 'hswish'),    # Start using SE and HSwish
        (40, 40, 5, 1, 6, 1, 'hswish'),    # Deep feature processing
        (40, 40, 5, 1, 6, 1, 'hswish'),    # More feature processing
        (40, 48, 5, 1, 3, 1, 'hswish'),    # Gradual channel expansion
        (48, 48, 5, 1, 3, 1, 'hswish'),    # Feature refinement
        (48, 96, 5, 2, 6, 1, 'hswish'),    # Final spatial reduction
        (96, 96, 5, 1, 6, 1, 'hswish'),    # High-level feature processing
        (96, 96, 5, 1, 6, 1, 'hswish'),    # Final feature processing
    ]
    
    # Main network
    for config in block_configs:
        x = Block(*config)(x)
    
    # Final feature extraction
    x = layers.Conv2D(96, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = HSwish()(x)
    x = layers.Conv2D(576, 1)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1280, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation('sigmoid')(x)
    
    model = Model(inputs, outputs)
    _initialize_weights(model)
    
    return model

def _initialize_weights(model):
    """
    - Random normal initialization for conv/dense layers
    - Zeros for biases
    - Ones/zeros for batch normalization
    """
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, layers.Dense)):
            layer.kernel_initializer = RandomNormal(mean=0.0, stddev=0.01)
            if layer.use_bias:
                layer.bias_initializer = 'zeros'
        elif isinstance(layer, layers.BatchNormalization):
            layer.gamma_initializer = 'ones'
            layer.beta_initializer = 'zeros'