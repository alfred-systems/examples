/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.transfer.api;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.TreeMap;

class LiteInferenceModel implements Closeable, ModelUtil {
  private static final int FLOAT_BYTES = 4;

  private final LiteModelWrapper modelWrapper;
  private final int numClasses;
  private int[] outputShape;

  LiteInferenceModel(LiteModelWrapper modelWrapper, int numClasses) {
    this.modelWrapper = modelWrapper;
    this.numClasses = numClasses;

    Interpreter interpreter = this.modelWrapper.getInterpreter();
    outputShape = interpreter.getOutputTensor(0).shape();
  }

  float[] runInference(ByteBuffer bottleneck, ByteBuffer[] modelParameters) {
    ByteBuffer predictionsBuffer = ByteBuffer.allocateDirect(numClasses * FLOAT_BYTES);
    predictionsBuffer.order(ByteOrder.nativeOrder());

    Map<Integer, Object> outputs = new TreeMap<>();
    outputs.put(0, predictionsBuffer);

    Object[] inputs = new Object[modelParameters.length + 1];
    inputs[0] = bottleneck;
    System.arraycopy(modelParameters, 0, inputs, 1, modelParameters.length);

    modelWrapper.getInterpreter().runForMultipleInputsOutputs(inputs, outputs);
    bottleneck.rewind();
    for (ByteBuffer buffer : modelParameters) {
      buffer.rewind();
    }
    predictionsBuffer.rewind();

    float[] predictions = new float[numClasses];
    for (int classIdx = 0; classIdx < numClasses; classIdx++) {
      predictions[classIdx] = predictionsBuffer.getFloat();
    }

    return predictions;
  }

  ByteBuffer runInference(ByteBuffer bottleneck, ByteBuffer[] modelParameters, String outputOpName) {

    ByteBuffer predictionsBuffer = ByteBuffer.allocateDirect(
      outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3]
    );
    predictionsBuffer.order(ByteOrder.nativeOrder());

    Map<Integer, Object> outputs = new TreeMap<>();
    outputs.put(0, predictionsBuffer);

    Object[] inputs = new Object[modelParameters.length + 1];
    inputs[0] = bottleneck;
    System.arraycopy(modelParameters, 0, inputs, 1, modelParameters.length);

    modelWrapper.getInterpreter().runForMultipleInputsOutputs(inputs, outputs);
    bottleneck.rewind();
    for (ByteBuffer buffer : modelParameters) {
      buffer.rewind();
    }
    predictionsBuffer.rewind();

    return predictionsBuffer;
  }

  boolean outputImage() {
    Interpreter interpreter = this.modelWrapper.getInterpreter();
    Tensor out_tenor = interpreter.getOutputTensor(0);
    DataType out_dtype = out_tenor.dataType();
    return DataType.UINT8 == out_dtype;
  }

  @Override
  public void close() {
    modelWrapper.close();
  }

  @Override
  public LiteModelWrapper getModelWrapper() {
    return this.modelWrapper;
  }

  @Override
  public void setOutputShape(int[] input_shape) {
    this.outputShape[0] = input_shape[0];
    this.outputShape[1] = input_shape[1];
    this.outputShape[2] = input_shape[2];
    this.outputShape[3] = 4;
  }
}
