import numpy as np
import tensorflow as tf
#from keras.saving import saving_lib
import tensorflow.keras as keras
    
def get_conv1D(model, idx, fs=20480):
    center_hz, bandwidth = model.layers[idx].get_weights()
    kernel_size, stride, padding = model.layers[1].kernel_size, model.layers[1].stride, model.layers[1].padding
    out_channels = len(center_hz)
    
    # time grid
    t = tf.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    t = tf.cast(t, tf.float32) / fs
    t = tf.reshape(t, [kernel_size, 1])  # [T,1]
    
    # broadcast params
    f0 = tf.reshape(center_hz, [1, out_channels])   # center frequency
    bw = tf.reshape(bandwidth, [1, out_channels])   # bandwidth
    
    # Gabor = sinusoid * Gaussian window
    pi = tf.constant(3.141592653589793, tf.float32)
    sinusoid = tf.cos(2 * pi * f0 * t)
    gaussian = tf.exp(- (pi * bw * t) ** 2)
    
    kernels = sinusoid * gaussian  # [T, C]
    
    # Conv1D kernel shape
    kernels = tf.reshape(kernels, [kernel_size, 1, out_channels]).numpy()
    
    kernels[np.abs(kernels)<0.1]=0
    
    conv1d = keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding=padding,
                           use_bias=False,
                           kernel_initializer=tf.constant_initializer(kernels), name="conv1D_gabor")
    return conv1d, kernels


@tf.keras.utils.register_keras_serializable()
class FakeQuantLayer(tf.keras.layers.Layer):
    def __init__(self, num_bits=16, min_val=-1.0, max_val=1.0, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.num_bits = num_bits

    def call(self, x):
        # simulate int16 (16-bit symmetric)
        return tf.quantization.fake_quant_with_min_max_vars(
            x, min=self.min_val, max=self.max_val, num_bits=self.num_bits
        )
    

def get_new_layer(old_layer, shape):
    # Get the config and recreate it
    new_layer = type(old_layer).from_config(old_layer.get_config())

    # Copy the weights
    new_layer.build(shape)  # Build layer with correct input shape
    new_layer.set_weights(old_layer.get_weights())
    return new_layer

def get_activation_model(model, layer_names):
    outputs = [model.get_layer(name).output for name in layer_names]
    return tf.keras.Model(inputs=model.input, outputs=outputs)
