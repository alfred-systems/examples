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
"""Adam optimizer implementation for transfer learning models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy.lib.function_base import gradient

import tensorflow as tf
import tensorflow.compat.v1 as tfv1


class Adam(object):
  """Adam optimizer configuration for transfer learning converter."""

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    self._learning_rate = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._eps = eps

  def optimizer_model_graph(self, parameter_shapes):
    """Generates a TFLite model that represents an optimizer step.

    The generated model inputs are current values of the trainable
    model parameters, followed by their gradients, and then by
    the current mutable optimizer state.

    The generated model outputs are the new values of the trainable
    parameters, followed by the updated mutable optimizer state.

    Args:
      parameter_shapes: list of model parameter shapes.

    Returns:
      TFLite optimizer model.
    """
    with tfv1.Session(graph=tf.Graph()) as sess:
      current_values = [
          tfv1.placeholder(tf.float32, shape, name=f"current_val_{i}")
          for i, shape in enumerate(parameter_shapes)
      ]
      gradients = [
          tfv1.placeholder(tf.float32, shape, name=F"grad_{i}")
          for i, shape in enumerate(parameter_shapes)
      ]
      clip_gradients = [tf.clip_by_value(g, -1., 1.) for g in gradients]
      ms = [
        tfv1.placeholder(tf.float32, shape, name=f'm_{i}')
        for i, shape in enumerate(parameter_shapes)
      ]
      vs = [
        tfv1.placeholder(tf.float32, shape, name=f'v_{i}')
        for i, shape in enumerate(parameter_shapes)
      ]
      step = tfv1.placeholder(tf.float32, (), name='step')

      new_values = []
      new_ms = []
      new_vs = []
      for cur_param, grad, m, v in zip(current_values, clip_gradients, ms, vs):
        m = (1 - self._beta1) * grad + self._beta1 * m
        v = (1 - self._beta2) * (grad**2) + self._beta2 * v
        mhat = m / (1 - self._beta1**(step + 1))
        vhat = v / (1 - self._beta2**(step + 1))
        new_param = cur_param - (
            self._learning_rate * mhat / (tf.sqrt(vhat) + self._eps))
        new_values.append(new_param)
        new_ms.append(m)
        new_vs.append(v)

      inputs = current_values + gradients + ms + vs + [step]
      outputs = new_values + new_ms + new_vs + [step + 1]
      return sess, inputs, outputs
  
  def py_optimizer_model_graph(self, parameter_shapes):
    """Generates a TFLite model that represents an optimizer step.

    The generated model inputs are current values of the trainable
    model parameters, followed by their gradients, and then by
    the current mutable optimizer state.

    The generated model outputs are the new values of the trainable
    parameters, followed by the updated mutable optimizer state.

    Args:
      parameter_shapes: list of model parameter shapes.

    Returns:
      TFLite optimizer model.
    """
    sess = tfv1.Session()
    current_values = [
        tfv1.placeholder(tf.float32, shape, name=f"current_val_{i}")
        for i, shape in enumerate(parameter_shapes)
    ]
    gradients = [
        tfv1.placeholder(tf.float32, shape, name=F"grad_{i}")
        for i, shape in enumerate(parameter_shapes)
    ]
    clip_gradients = [tf.clip_by_value(g, -1., 1.) for g in gradients]
    ms = [
      tfv1.placeholder(tf.float32, shape, name=f'm_{i}')
      for i, shape in enumerate(parameter_shapes)
    ]
    vs = [
      tfv1.placeholder(tf.float32, shape, name=f'v_{i}')
      for i, shape in enumerate(parameter_shapes)
    ]
    step = tfv1.placeholder(tf.float32, (), name='step')

    new_values = []
    new_ms = []
    new_vs = []
    for cur_param, grad, m, v in zip(current_values, clip_gradients, ms, vs):
      m = (1 - self._beta1) * grad + self._beta1 * m
      v = (1 - self._beta2) * (grad**2) + self._beta2 * v
      mhat = m / (1 - self._beta1**(step + 1))
      vhat = v / (1 - self._beta2**(step + 1))
      new_param = cur_param - (
          self._learning_rate * mhat / (tf.sqrt(vhat) + self._eps))
      new_values.append(new_param)
      new_ms.append(m)
      new_vs.append(v)

    inputs = current_values + gradients + ms + vs + [step]
    outputs = new_values + new_ms + new_vs + [step + 1]
    return sess, inputs, outputs
  
  def generate_optimizer_model(self, parameter_shapes):
    """Generates a TFLite model that represents an optimizer step.

    The generated model inputs are current values of the trainable
    model parameters, followed by their gradients, and then by
    the current mutable optimizer state.

    The generated model outputs are the new values of the trainable
    parameters, followed by the updated mutable optimizer state.

    Args:
      parameter_shapes: list of model parameter shapes.

    Returns:
      TFLite optimizer model.
    """
    session, inputs, outputs = self.optimizer_model_graph(parameter_shapes)
    converter = tfv1.lite.TFLiteConverter.from_session(
        session, inputs, outputs)
    return converter.convert()


def test_adam():
  tf.compat.v1.disable_eager_execution()

  adam = Adam()
  session, inputs, outputs = adam.py_optimizer_model_graph([(), (), ()])
  out_val = session.run(outputs, feed_dict={
    inputs[0]: 0.05,
    inputs[1]: 0.21,
    inputs[2]: 0.65,
    
    inputs[3]: 0.192,
    inputs[4]: -0.8104,
    inputs[5]: -0.0623,

    inputs[6]: 0.0,
    inputs[7]: 0.0,
    inputs[8]: 0.0,
    inputs[9]: 0.0,
    inputs[10]: 0.0,
    inputs[11]: 0.0,
    
    inputs[12]: 0.0,
  })

  for i, o in enumerate(out_val):
    print(i, o)


if __name__ == "__main__":
    test_adam()