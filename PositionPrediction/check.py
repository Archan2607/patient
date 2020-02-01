import tensorflow as tf
from tensorflow.python.platform import gfile

# for tensorboard
with tf.Session() as sess:
    model_filename = "model_final_checkpoints/saved_model.pb"
    with gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR = "__tb"
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

"""
Then from terminal run
tensorboard --logdir __tb
            
            OR
            
you can also go for netron, check below link
https://lutzroeder.github.io/netron/
"""

# writing node name to the text file
model_path = "model_final_checkpoints/saved_model.pb"
f = tf.gfile.GFile(model_path, "rb")
gd = graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
with open("somefile.txt", "wb") as f:
    for node in graph_def.node:
        f.write(node.name.encode("utf-8") + "\n".encode("utf-8"))
