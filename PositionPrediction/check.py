# error
# Input 0 of node global_step/Assign was passed int64 from global_step:0 incompatible with expected int64_ref.
import tensorflow as tf

model_path = "model_dir/saved_model.pb"

# read graph definition
f = tf.gfile.GFile(model_path, 'rb')
gd = graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())


# fix nodes
for node in graph_def.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']

tf.import_graph_def(graph_def, name='')
# tf.train.write_graph(graph_def, './', 'good_frozen.pb', as_text=False)
# tf.train.write_graph(graph_def, './', 'good_frozen.pbtxt', as_text=True)

# import graph into session
# tf.import_graph_def(graph_def, name='')
######################################################################################################

# import tensorflow as tf
# from tensorflow.python.platform import gfile
# with tf.Session() as sess:

# 	 model_filename ='frozen_dir/model.pb'
#	 with gfile.FastGFile(model_filename, 'rb') as f:
#		 graph_def = tf.GraphDef()
#		 graph_def.ParseFromString(f.read())
#		 g_in = tf.import_graph_def(graph_def)
# LOGDIR='__tb'
# train_writer = tf.summary.FileWriter(LOGDIR)
# train_writer.add_graph(sess.graph)

## Then from terminal run
## tensorboard --logdir __tb

import tensorflow as tf

# def load_graph(model_file):
#     """
#   Code from v1.6.0 of Tensorflow's label_image.py example
#   """
#     graph_ = tf.Graph()
#     graph_def = tf.GraphDef()
#     with open(model_file, "rb") as f:
#         graph_def.ParseFromString(f.read())
#     with graph_.as_default():
#         tf.import_graph_def(graph_def)
#     return graph_
#
#
# graph = load_graph('model_dir/saved_model.pb')
