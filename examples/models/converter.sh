#!/bin/bash

# Convert existing PPO model file for TensorFlow.js
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    . \
    ./baseline/
