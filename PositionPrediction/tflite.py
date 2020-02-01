import tensorflow as tf

model_dir = "model_final_checkpoints"
pbtxt_filename = "saved_model.pbtxt"

# We retrieve our checkpoint fullpath
checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# We precise the file fullname of our freezed graph
absolute_model_dir = "/".join(input_checkpoint.split("/")[:-1])
output_graph = absolute_model_dir + "/" + pbtxt_filename

# creating the .pbtxt file
# using seperrate session
sess = tf.Session()

tf.train.write_graph(
    graph_or_graph_def=sess.graph_def,
    logdir=model_dir,
    name=pbtxt_filename,
    as_text=True,
)

# closing the session
sess.close()


# freezing the model
# creating new session
graph = tf.Graph()
with graph.as_default():

    meta_path = input_checkpoint + ".meta"
    saver = tf.train.import_meta_graph(meta_path)
    with tf.Session() as sess:

        saver.restore(sess, input_checkpoint)
        output_node_names = "training_loss"
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), output_node_names.split(",")
        )
