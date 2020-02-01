import tensorflow as tf

model_dir = "model_final_checkpoints"

# We retrieve our checkpoint fullpath
checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# We precise the file fullname of our freezed graph
absolute_model_dir = "/".join(input_checkpoint.split("/")[:-1])
output_graph = absolute_model_dir + "/saved_model.pb"

"""
ERROR(Resolved):

tensorflow.python.framework.errors_impl.InvalidArgumentError: 
Input 0 of node save/Assign was passed int64 from global_step_2:0 
incompatible with expected int64_ref.
"""

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    output_graph,
    input_shapes={"tree-0": [5189, 13]},
    input_arrays=["tree-0"],
    output_arrays=["Mean"],
)

converter.allow_custom_ops = True

# convert model to .tflite FORMAT
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
