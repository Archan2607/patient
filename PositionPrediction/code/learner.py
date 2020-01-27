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

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


class Learner:
    """
    Class contains function to train the model on train data.
    """

    def __init__(self, n_estimators, min_samples_split, min_samples_leaf, random_state, model_folder, model_name):
        self.n_estimators = n_estimators  # number of tree hyperparameter
        self.min_samples_split = min_samples_split  # minimum samples to split hyperparameter
        self.min_samples_leaf = min_samples_leaf  # minimum samples leaf hyperparameter
        self.random_state = random_state  # random state hyperparameter
        self.model_folder = model_folder  # folder to save the models
        self.model_name = model_name  # name of the model

    def make_input_fn(self, X, y, NUM_EXAMPLES, n_epochs=None, shuffle=True):

        def input_fn():
            dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
            if shuffle:
                dataset = dataset.shuffle(NUM_EXAMPLES)
            # For training, cycle thru dataset as many times as need (n_epochs=None).
            dataset = dataset.repeat(n_epochs)
            # In memory training doesn't use batching.
            dataset = dataset.batch(NUM_EXAMPLES)
            return dataset

        return input_fn

    def getFeature(self, feature_names):

        feature_columns = []
        for names in feature_names:
            feature_columns.append(tf.feature_column.numeric_column(names, dtype=tf.float32))

        return feature_columns

    def serving_input_receiver_fn(self, hyperparameters=None):

        features = {'x': tf.placeholder(
            shape=[None, None],
            dtype=tf.float32)
        }
        return tf.estimator.export.ServingInputReceiver(
            receiver_tensors=features,
            features=features
        )

    def train_model(self, X_train, Y_train):
        """
        Function to train the model on train data and save the trained model in model folder
        """
        # NUM_EXAMPLES = len(Y_train)
        print(X_train)
        for idx in range(len(Y_train)):
            Y_train[idx] = Y_train[idx] - 1

        Y_train = pd.DataFrame({'manual_position': list(Y_train)})
        Y_train = Y_train['manual_position']

        x = X_train.to_dict('list')
        for key in x:
            x[key] = np.expand_dims(np.array(x[key], dtype=np.float32), axis=1)

        y = Y_train.values
        y = np.expand_dims(y, axis=1)

        feature_names = ["left_sensors_pct", 'plank_1_std', 'plank_2_std', 'plank_3_std', 'plank_4_std', 'y_errors',
                         'plank_3_dev_bucket',
                         'plank_4_dev_bucket', 'plank_1_com_y', 'plank_2_com_y', 'plank_3_com_y', 'plank_4_com_y',
                         'plank_4_wrt_3_2']

        features = self.getFeature(feature_names)

        # Training and evaluation input functions.
        # train_input_fn = self.make_input_fn(X_train, Y_train, NUM_EXAMPLES, n_epochs=100)

        # linear_est = tf.estimator.BoostedTreesClassifier(features,
        # n_batches_per_layer=200, n_classes=5)

        # Train model.
        # linear_est.train(train_input_fn, max_steps=100)

        # define input function
        # train_input_fn = self.make_input_fn(
        #     X_train,
        #     Y_train,
        #     NUM_EXAMPLES,
        #     n_epochs=100
        # )

        params = tensor_forest.ForestHParams(
            feature_colums=features,
            # N_ESTIMATORS=100,
            # MIN_SAMPLES_SPLIT=25,
            # MIN_SAMPLES_LEAF=5,
            # RANDOM_STATE=12,
            num_trees=10,
            max_nodes=1000,
            # regression=True,
            num_classes=5,
            num_features=13
        )

        # build the graph
        graph_builder_class = tensor_forest.RandomForestGraphs

        est = random_forest.TensorForestEstimator(
            params, graph_builder_class=graph_builder_class,
            model_dir='model_dir')

        # define an input function
        train_input_fn = numpy_io.numpy_input_fn(
            x=x,
            y=y,
            batch_size=32,
            num_epochs=1000,
            shuffle=True,
        )

        est.fit(
            input_fn=train_input_fn,
        )

        # est.export_savedmodel('saved_model', self.serving_input_receiver_fn)
        return 1, 2

