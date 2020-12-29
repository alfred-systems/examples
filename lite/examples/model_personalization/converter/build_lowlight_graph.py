from loguru import logger
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.engine.input_layer import InputLayer
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter


class Preprocess(tf.keras.layers.Layer):

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = x / 255
        return x
        

class LLFilter(tf.keras.layers.Layer):

    def __init__(self, a=0.05, b=0.21, c=0.65):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
    
    def build(self, input_shape):
        self.Y_w = tf.Variable([0.29900, 0.58700, 0.11400], trainable=False)
        # self.Y_w = self._Y_w[None, None, None, :]

        self.A = tf.Variable(self.a, trainable=True)
        self.B = tf.Variable(self.b, trainable=True)
        self.C = tf.Variable(self.c, trainable=True)
        
        # self._A = tf.Variable(a, trainable=True)
        # self._B = tf.Variable(b, trainable=True)
        # self._C = tf.Variable(c, trainable=True)

    def mapping(self, x, default=False):
        if default:
            r = tf.math.log(x * 5.0 + tf.clip_by_value(self._A, 1e-6, 1e10)) * self._B + self._C
        else:
            r = tf.math.log(x * 5.0 + tf.clip_by_value(self.A, 1e-6, 1e10)) * self.B + self.C
        return tf.clip_by_value(r, 0, 255)
    
    # @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 3], dtype=tf.float32, name="fp_img")])
    def call(self, x):
        lum = tf.reduce_sum((x * self.Y_w), keepdims=True, axis=1)
        adj_lum = self.mapping(lum)
        # adj_lum = (adj_lum
        return adj_lum


def sobel_loss(fp_img, _):
    edges = tf.image.sobel_edges(fp_img)
    return -tf.math.reduce_sum(edges)


def demo():
    base = bases.MobileNetV2Base(image_size=224)
    head = tf.keras.Sequential([
        layers.Flatten(input_shape=(7, 7, 1280)),
        layers.Dense(
            units=32,
            activation='relu',
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
        layers.Dense(
            units=4,
            activation='softmax',
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
    ])

    # Optimizer is ignored by the converter! See docs.
    head.compile(loss='categorical_crossentropy', optimizer='sgd')

    converter = TFLiteTransferConverter(4,
                                        base,
                                        heads.KerasModelHead(head),
                                        optimizers.SGD(3e-2),
                                        train_batch_size=20)

    converter.convert_and_save('custom_keras_model')


def export_transfer_model(h=256, w=256):
    # tf.compat.v1.disable_eager_execution()
    # sess = tf.compat.v1.Session()

    # with sess.as_default():
    base = bases.ImagePreprocessBase((1, 256, 256, 3), type=None)
    
    # z = tf.zeros([1, 256, 256, 3], dtype=tf.float32)
    # head_model = tf.keras.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=[None, None, 3]),
    #     LLFilter(),
    # ])
    # head_model(z)
    # head_model.compile(loss=sobel_loss, optimizer='adam')

    # # Optimizer is ignored by the converter! See docs.
    
    # # sess.run(tf.compat.v1.global_variables_initializer())
    # head = heads.KerasModelHead(head_model)
    head = heads.LowLightFilterHead(1, [h, w, 3])

    converter = TFLiteTransferConverter(4,
                                        base,
                                        head,
                                        optimizers.Adam(1e-3),
                                        train_batch_size=1)
    converter.convert_and_save('low_light_filter_head')


if __name__ == "__main__":
    
    with logger.catch(reraise=True):
        # f = LLFilter()
        # z = tf.zeros([1, 64, 64, 3])
        # f(z)
        export_transfer_model()
        # demo()
