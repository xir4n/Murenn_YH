from filterlayers import *
from tf_quantize import *
from utils import *


if __name__ == "__main__":

    input_layer = tf.keras.layers.Input(shape=(30720,1))
    gabor_layer = GaborConv1D(out_channels=4, kernel_size=101, stride=64, fs=20480)(input_layer)

    gabor_feature = tf.keras.models.Model(inputs=input_layer, outputs=gabor_layer)
    gabor_feature.summary()

    gaborConv, _ = get_conv1D(gabor_feature, 1, fs=20480)
    conv_layer = gaborConv(input_layer)
    conv_feature = tf.keras.models.Model(inputs=input_layer, outputs=conv_layer)
    conv_feature.summary()

    outfolder=""
    model_path=outfolder+"gabor_feature.keras"
    conv_feature.save(model_path)
    convert_model(outfolder, model_path, "gabor_feature")