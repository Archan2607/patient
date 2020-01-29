import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

# dir = os.path.dirname(os.path.realpath(__file__))
#
#
# def freeze_graph(model_dir):
#
#     # We retrieve our checkpoint fullpath
#     checkpoint = tf.train.get_checkpoint_state(model_dir)
#     input_checkpoint = checkpoint.model_checkpoint_path
#
#     # We precise the file fullname of our freezed graph
#     absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
#     output_graph = absolute_model_dir + "/frozen_model.pb"
#
#     # We clear devices to allow TensorFlow to control on which device it will load operations
#     clear_devices = True
#
#     sess = tf.Session()
#
#     saver = tf.train.import_meta_graph('model_dir/model.ckpt-482.meta')
#
#     # saver.restore(sess, input_checkpoint)
#     # output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
#     # print(output_node_names)
#
#     # We start a session using a temporary fresh Graph
#     # with tf.Session() as sess:
#     #     # We import the meta graph in the current default Graph
#     #     # saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
#     #
#     #     saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
#     #
#     #     # We restore the weights
#     #     saver.restore(sess, input_checkpoint)
#     #
#     #     # We use a built-in TF helper to export variables to constants
#     #     output_graph_def = tf.graph_util.convert_variables_to_constants(
#     #         sess,  # The session is used to retrieve the weights
#     #         tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
#     #         output_node_names.split(",")  # The output node names are used to select the usefull nodes
#     #     )
#     #
#     #     # Finally we serialize and dump the output graph to the filesystem
#     #     with tf.gfile.GFile(output_graph, "wb") as f:
#     #         f.write(output_graph_def.SerializeToString())
#     #     print("%d ops in the final graph." % len(output_graph_def.node))
#     #
#     # return output_graph_def
#
#
# freeze_graph('model_dir')
init = tf.global_variables_initializer()
graph = tf.Graph()
with graph.as_default():
    saver = tf.train.import_meta_graph("model_dir/model.ckpt-482.meta")
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "model_dir/model.ckpt-482")
        output_node_names = 'init'# [n.name for n in tf.get_default_graph().as_graph_def().node]
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names)

tf.lite.TFLiteConverter.from_frozen_graph(
    'frozen_dir/saved_model.pb',
    input_arrays='global_step/Initializer/zeros',
    output_arrays='init'
)