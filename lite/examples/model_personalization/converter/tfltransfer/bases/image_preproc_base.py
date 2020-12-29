from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.compat import v1 as tfv1

# pylint: disable=g-bad-import-order
from tfltransfer.bases import quantizable_base
# pylint: enable=g-bad-import-order


class ImagePreprocessBase(quantizable_base.QuantizableBase):
  """Base model configuration that reads a specified SavedModel.

  The SavedModel should contain a signature that converts
  samples to bottlenecks. This is assumed by default to be
  the main serving signature, but this can be configured.
  """

  def __init__(self,
               bottleneck_shape,
               type=None,
               tag=tf.saved_model.SERVING,
               signature_key='serving_default',
               quantize=False,
               representative_dataset=None):
    """Constructs a base model from a SavedModel.

    Args:
      model_dir: path to the SavedModel to load.
      tag: MetaGraphDef tag to be used.
      signature_key: signature name for the forward pass.
      quantize: whether the model weights should be quantized.
      representative_dataset: generator that yields representative data for full
        integer quantization. If None, hybrid quantization is performed.
    """
    super(ImagePreprocessBase, self).__init__(quantize, representative_dataset)
    self._tag = tag
    self._signature_key = signature_key

    self._bottleneck_shape = bottleneck_shape
    self._type = type

  def prepare_converter_norm_uint8(self):
    """Prepares an initial configuration of a TFLiteConverter."""
    with tf.Graph().as_default(), tfv1.Session() as sess:
      bottleneck_shape = self._bottleneck_shape
      image = tfv1.placeholder(tf.uint8, bottleneck_shape, 'image')
      norm_image = tf.cast(image, tf.float32) / 255
      
      converter = tfv1.lite.TFLiteConverter.from_session(
          sess, [image], [norm_image])
      converter.inference_input_type = tf.uint8
      return converter
  
  def prepare_converter_no_op(self):
    """Prepares an initial configuration of a TFLiteConverter."""
    with tf.Graph().as_default(), tfv1.Session() as sess:
      bottleneck_shape = self._bottleneck_shape
      image = tfv1.placeholder(tf.float32, bottleneck_shape, 'image')
      _image = tf.identity(image)
      
      converter = tfv1.lite.TFLiteConverter.from_session(
          sess, [image], [_image])
      return converter
    
  def prepare_converter(self):
    if self._type is None:
      return self.prepare_converter_no_op()
    elif self._type == 'norm':
      return self.prepare_converter_norm_uint8()

  def bottleneck_shape(self):
    """Reads the shape of the bottleneck produced by the model."""
    return self._bottleneck_shape
