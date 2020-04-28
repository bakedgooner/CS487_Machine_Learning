import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class CNN(object):
    def __init__(self,
                 batchsize=100,
                 epochs=20,
                 learning_rate=1e-4,
                 dropout_rate=0.5,
                 shuffle=True,
                 random_seed=None):
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle

        g = tf.Graph()

        with g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=g)

    def build(self):
        tf_X = tf.placeholder(tf.float32, shape=[None, 784], name="tf_X")
        tf_y = tf.placeholder(tf.int32, shape=[None], name="tf_y")
        is_train = tf.placeholder(tf.bool, shape=(), name="is_train")

        tf_X_img = tf.reshape(
            tf_X, shape=[-1, 28, 28, 1], name="input_X_2dimgs")
        tf_y_onehot = tf.one_hot(
            indices=tf_y, depth=10, dtype=tf.float32, name="input_y_onehot")

        # Layer 1
        l1 = tf.layers.conv2d(tf_X_img, kernel_size=(3, 3), strides=(
            1, 1), padding='valid', filters=4, activation=tf.nn.relu)
        l1_pool = tf.layers.max_pooling2d(l1, pool_size=(2, 2), strides=(2, 2))

        # Layer 2
        l2 = tf.layers.conv2d(l1_pool, kernel_size=(3, 3), strides=(
            3, 3), padding='valid', filters=2, activation=tf.nn.relu)
        l2_pool = tf.layers.max_pooling2d(l2, pool_size=(4, 4), strides=(4, 4))

        # Layer 3
        input_shape = l2_pool.get_shape().as_list()
        n_input = np.prod(input_shape[1:])
        l2_pool_flat = tf.reshape(l2_pool, shape=[-1, n_input])
        l3 = tf.layers.dense(l2_pool_flat, 10, activation=tf.nn.relu)

        # Preds
        preds = {
            "probs": tf.nn.softmax(l3, name="probs"),
            "labels": tf.cast(tf.argmax(l3, axis=1), tf.int32, name="labels")
        }

        # cross entropy loss
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=l3, labels=tf_y_onehot), name="ce_loss")

        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(ce_loss, name="train_optimizer")

        # Accuracy
        correct = tf.equal(preds["labels"], tf_y, name="correct")
        acc = tf.reduce_mean(tf.cast(correct, tf.float32), name="acc")

    def batches(self, X, y, batch_size=100, shuffle=False, random_seed=None):
        index = np.arange(y.shape[0])
        if shuffle:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(index)
            X = X[index]
            y = y[index]

        for i in range(0, X.shape[0], batch_size):
            yield (X[i: i + batch_size, :], y[i: i + batch_size])

    def train(self, train_set, validation_set=None, initialize=True):
        if initialize:
            self.sess.run(self.init_op)

        self.train_cost_ = []
        X_data = np.array(train_set[0])
        y_data = np.array(train_set[1])

        for epoch in range(1, self.epochs + 1):
            batch = self.batches(X_data, y_data, shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_X, batch_y) in enumerate(batch):
                feed = {"tf_X:0": batch_X,
                        "tf_y:0": batch_y, "is_train:0": True}
                loss, _ = self.sess.run(
                    ["ce_loss:0", "train_optimizer"], feed_dict=feed)
                avg_loss += loss
            print("Epoch %02d: Training Average Loss: ""%7.3f" %
                  (epoch, avg_loss), end=" ")
            if validation_set is not None:
                feed = {"tf_X:0": batch_X,
                        "tf_y:0": batch_y, "is_train:0": False}
                valid_acc = self.sess.run("acc:0", feed_dict=feed)
                print("Validation Accuracy: %7.3f" % valid_acc)
            else:
                print()

    def predict(self, X_test, return_prob=False):
        feed = {"tf_X:0": X_test, "is_train:0": False}
        if return_prob:
            return self.sess.run("probs:0", feed_dict=feed)
        else:
            return self.sess.run("labels:0", feed_dict=feed)

    def save(self, epoch, path="./model"):
        if not os.path.isdir(path):
            os.makedirs(path)
        print("Saving Model")
        self.saver.save(self.sess, os.path.join(
            path, "model.ckpt"), global_step=epoch)

    def load(self, epoch, path):
        print("Loading Model")
        self.saver.restore(self.sess, os.path.join(
            path, "model.ckpt-%d" % epoch))
