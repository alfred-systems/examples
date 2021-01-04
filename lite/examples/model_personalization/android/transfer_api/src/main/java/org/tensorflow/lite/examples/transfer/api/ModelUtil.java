package org.tensorflow.lite.examples.transfer.api;

import org.tensorflow.lite.Interpreter;

public interface ModelUtil {

    default void resizeInput(int input_id, int[] shape) {
        LiteModelWrapper modelWrapper = this.getModelWrapper();
        Interpreter interpreter = modelWrapper.getInterpreter();
        interpreter.resizeInput(input_id, shape);
        this.setOutputShape(shape);
    }

    default void setOutputShape(int[] input_shape) {

    }

    LiteModelWrapper getModelWrapper();
}
