# import os
# import pickle
# from tensorflow.python.ops import resources
# from sklearn.ensemble import RandomForestClassifier

from __future__ import print_function
import numpy as np

from tensorflow.python.estimator.inputs import numpy_io
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.python.ops import resources
import config

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


class Learner:
    """
    Class contains function to train the model on train data.
    """

    def __init__(
        self,
        n_estimators,
        min_samples_split,
        min_samples_leaf,
        random_state,
        model_folder,
        model_name,
    ):
        self.n_estimators = n_estimators  # number of tree hyperparameter
        self.min_samples_split = (
            min_samples_split  # minimum samples to split hyperparameter
        )
        self.min_samples_leaf = min_samples_leaf  # minimum samples leaf hyperparameter
        self.random_state = random_state  # random state hyperparameter
        self.model_folder = model_folder  # folder to save the models
        self.model_name = model_name  # name of the model

    def getFeature(self, feature_names):

        feature_columns = []
        for names in feature_names:
            feature_columns.append(
                tf.feature_column.numeric_column(names, dtype=tf.float32)
            )

        return feature_columns

    def next_batch(self, num, data, labels):
        """
        Return a total of `num` random samples and labels.
        """
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    def train_model(self, X_train, Y_train):

        # input features of size(None, 13)
        x_train = X_train.values

        # output label of size(None)
        y_train = Y_train.values

        # creating a placeholder for Input and Target data
        X = tf.placeholder(tf.float32, shape=[None, 13])
        Y = tf.placeholder(tf.int8, shape=[None])

        feature_names = config.FEATURE_NAMES
        features = self.getFeature(feature_names)

        params = tensor_forest.ForestHParams(
            # N_ESTIMATORS=100,
            # MIN_SAMPLES_SPLIT=25,
            # MIN_SAMPLES_LEAF=5,
            # RANDOM_STATE=12,
            feature_colums=features,
            num_trees=10,
            max_nodes=1000,
            num_classes=6,
            num_features=13,
        ).fill()

        # Build the Random Forest
        forest_graph = tensor_forest.RandomForestGraphs(params)

        # Get training graph and loss
        train_op = forest_graph.training_graph(X, Y)
        loss_op = forest_graph.training_loss(X, Y)

        saver = tf.train.Saver()

        init = tf.group(
            tf.global_variables_initializer(),
            resources.initialize_resources(resources.shared_resources()),
        )

        batch_size = 1000

        sess = tf.Session()

        sess.run(init)
        for i in range(100):
            Xtr, Ytr = self.next_batch(batch_size, x_train, y_train)

            # Feed actual data to the train operation
            sess.run([loss_op, train_op], feed_dict={X: Xtr, Y: Ytr})

            # Create a checkpoint in every iteration
            saver.save(sess, "model_checkpoints/model_iter", global_step=i)

        # Save the final model
        saver.save(sess, "model_final_checkpoints/model_final")

    """
    Below code might come handy.
    training done using estimator. 
    """

    # def train_model(self, X_train, Y_train):
    #     """
    #     Function to train the model on train data and save the trained model in model folder
    #     """
    #
    #     for idx in range(len(Y_train)):
    #         Y_train[idx] = Y_train[idx] - 1
    #
    #     Y_train = pd.DataFrame({"manual_position": list(Y_train)})
    #     Y_train = Y_train["manual_position"]
    #
    #     x = X_train.to_dict("list")
    #     for key in x:
    #         x[key] = np.expand_dims(np.array(x[key], dtype=np.float32), axis=1)
    #
    #     y = Y_train.values
    #     y = np.expand_dims(y, axis=1)
    #
    #     feature_names = [
    #         "left_sensors_pct",
    #         "plank_1_std",
    #         "plank_2_std",
    #         "plank_3_std",
    #         "plank_4_std",
    #         "y_errors",
    #         "plank_3_dev_bucket",
    #         "plank_4_dev_bucket",
    #         "plank_1_com_y",
    #         "plank_2_com_y",
    #         "plank_3_com_y",
    #         "plank_4_com_y",
    #         "plank_4_wrt_3_2",
    #     ]
    #
    #     features = self.getFeature(feature_names)
    #
    #     params = tensor_forest.ForestHParams(
    #         feature_colums=features,
    #         # N_ESTIMATORS=100,
    #         # MIN_SAMPLES_SPLIT=25,
    #         # MIN_SAMPLES_LEAF=5,
    #         # RANDOM_STATE=12,
    #         num_trees=10,
    #         max_nodes=1000,
    #         # regression=True,
    #         num_classes=5,
    #         num_features=13,
    #     )
    #
    #     # build the graph
    #     graph_builder_class = tensor_forest.RandomForestGraphs
    #
    #     est = random_forest.TensorForestEstimator(
    #         params, graph_builder_class=graph_builder_class
    #     )
    #
    #     # define an input function
    #     train_input_fn = numpy_io.numpy_input_fn(
    #         x=x, y=y, batch_size=32, num_epochs=1000, shuffle=True,
    #     )
    #
    #     est.fit(input_fn=train_input_fn,)
    #
    #     sess = tf.Session()
    #     saver = tf.train.Saver()
    #
    #     saver.save(sess, )
    #     return
