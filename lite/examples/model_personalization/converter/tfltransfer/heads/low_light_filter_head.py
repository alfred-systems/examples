# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Head model configuration for simple softmax classifiers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1


class LowLightFilterHead(object):
  """Head model configuration for a fixed classifier model architecture.

  This configuration does not require defining a custom model.
  It can be used when the head model should be a simple linear
  classifier: one fully-connected layer with softmax activation
  and cross-entropy loss function.

  This configuration can work without Flex runtime.
  """

  def __init__(self, train_batch_size, input_shape):
    """Constructs a SoftmaxClassifierHead instance.

    Args:
      train_batch_size: batch size to be used during training.
      input_shape: shape of the bottleneck inputs to the model.
      num_classes: number of classes for the target classification task.
      l2_reg: lambda parameter for L2 weights regularization. Default is no
        regularization.
    """
    self._train_batch_size = train_batch_size
    self._input_shape = tuple(input_shape)

  def predict(self, bottleneck, scope='head'):
    """Appends the serving signature of the model to the current graph.

    Bottleneck tensor is connected as an input to the added model.
    All model variables are converted to placeholders and returned
    in a list.

    Args:
      bottleneck: tensor in the current graph to be connected as input.
      scope: name of the scope to load the model into.

    Returns:
      (head model output tensor, list of variable placeholders)
    """
    logits, variables, _ = self._mapping(bottleneck, scope)
    predictions = tf.cast(tf.clip_by_value(logits * 255, 0, 255), tf.uint8)
    predictions_g = tf.cast(tf.clip_by_value(logits * 255 * 1.2, 0, 255), tf.uint8)
    predictions_a_pad = tf.cast(tf.clip_by_value(logits * 255 * 99, 0, 255), tf.uint8)
    
    predictions = tf.concat([
      predictions,
      predictions_g,
      predictions,
      predictions_a_pad,
    ], axis=-1)
    # predictions = tf.pad(
    #   predictions, 
    #   [[0, 0], [0, 0], [0, 0], [0, 2]], 
    #   constant_values=1
    # )
    # predictions = tf.pad(
    #   predictions, 
    #   [[0, 0], [0, 0], [0, 0], [0, 1]], 
    #   constant_values=255
    # )
    return predictions, variables

  def train(self, bottleneck, labels, scope='head'):
    """Appends the train signature of the model to the current graph.

    Bottleneck and labels tensors are connected as inputs.
    All model variables are converted to placeholders and returned
    in a list.

    Args:
      bottleneck: tensor containing input bottlenecks.
      labels: tensor containing one-hot ground truth labels.
      scope: name of the scope to load the model into.

    Returns:
      (loss tensor, list of variable gradients, list of variable placeholders)
    """
    logits, train_variables, flat_bottleneck = self._mapping(bottleneck, scope)
    with tf.name_scope(scope + '/loss'):
      loss = self._sobel_loss(logits) * -1
    
    with tf.name_scope(scope + '/backprop'):
      gradients = tf.gradients(
          loss, train_variables, stop_gradients=train_variables)
    return loss, gradients, train_variables

  def py_train(self, bottleneck, labels, scope='head'):
    """
    Tensorflow 1.0 version of training loop for verifiy result correctness
    Args:
      bottleneck: tensor containing input bottlenecks.
      labels: tensor containing one-hot ground truth labels.
      scope: name of the scope to load the model into.

    Returns:
      (loss tensor, list of variable gradients, list of variable placeholders)
    """
    logits, train_variables, flat_bottleneck = self._mapping(bottleneck, scope, use_var=True)
    with tf.name_scope(scope + '/loss'):
      loss = self._sobel_loss(logits) * -1
    
    with tf.name_scope(scope + '/backprop'):
      # gradients = tf.gradients(
      #     loss, train_variables, stop_gradients=train_variables)
      adam = tfv1.train.AdamOptimizer(learning_rate=0.01)
      grads = adam.compute_gradients(loss, var_list=train_variables)
      optimize_op = adam.apply_gradients(grads)
    return loss, optimize_op, logits, train_variables

  def _mapping(self, bottleneck, scope, use_var=False):
    """Appends the forward pass of the model."""
    with tfv1.variable_scope(scope):
      if use_var:
        a = tf.Variable(0.05, trainable=True)
        b = tf.Variable(0.2, trainable=True)
        c = tf.Variable(0.65, trainable=True)
      else:
        a = tfv1.placeholder(
            tf.float32,
            shape=(),
            name='placeholder_a')
        b = tfv1.placeholder(
            tf.float32, shape=(), name='placeholder_b')
        c = tfv1.placeholder(
            tf.float32, shape=(), name='placeholder_c')
      
      g = tf.reduce_sum(bottleneck * tf.constant([[[[0.29900, 0.58700, 0.11400]]]]), axis=-1, keepdims=True)
      r = tf.math.log(g * 5.0 + tf.clip_by_value(a, 1e-6, 1e10)) * b + c
      r = tf.clip_by_value(r, 0, 255)
      return r, [a, b, c], bottleneck
  
  def _sobel_loss(self, x):
    sobel_x = tf.constant([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
      ],
      tf.float32
    )
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    filtered_x = tf.nn.conv2d(
      x,
      sobel_x_filter,
      strides=[1, 1, 1, 1],
      padding='SAME'
    )
    filtered_y = tf.nn.conv2d(
      x,
      sobel_y_filter,
      strides=[1, 1, 1, 1],
      padding='SAME'
    )

    filtered_x = tf.clip_by_value(filtered_x, -.5, .5)
    filtered_y = tf.clip_by_value(filtered_y, -.5, .5)
    
    x_edge = tf.reduce_mean(tf.abs(filtered_x))
    y_edge = tf.reduce_mean(tf.abs(filtered_y))
    return x_edge + y_edge

  def generate_initial_params(self):
    """Constructs a TF function that computes initial parameter values.

    The function accepts a single scalar input that should always be
    zero. Without this input, TFLiteConverter eagerly converts all
    tf.fill instances into constants, instead of emitting Fill ops.

    Returns:
      TensorFlow function that returns initial model parameter values.
    """

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def model_func(zero):
      a = tf.constant(0.2)
      b = tf.constant(0.25)
      c = tf.constant(0.4)
      # a = tf.constant(0.05)
      # b = tf.constant(0.21)
      # c = tf.constant(0.65)
      return a, b, c

    return model_func

  def input_shape(self):
    """Returns the model input shape."""
    return self._input_shape

  def train_requires_flex(self):
    """Whether the generated training model requires Flex support."""
    return True


def test_py_train():
  tf.compat.v1.disable_eager_execution()
  sess = tf.compat.v1.Session()

  with sess.as_default():
    dummy_img_holder = tfv1.placeholder(shape=[1, 256, 256, 3], dtype=tf.float32, name='input')
    ll_filter = LowLightFilterHead(1, [256, 256, 3])
    loss, optimize_op, logits, train_variables = ll_filter.py_train(dummy_img_holder, None)
    sess.run(tf.compat.v1.global_variables_initializer())
    
    fake_img = np.random.uniform(low=0.0, high=1.0, size=[1, 256, 256, 3]).astype(np.float32)

    for _ in range(10):
      loss_val, _ = sess.run([loss, optimize_op], feed_dict={
        dummy_img_holder: fake_img
      })
      print('loss: ', loss_val)


if __name__ == "__main__":
    test_py_train()