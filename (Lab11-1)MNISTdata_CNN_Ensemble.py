import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
import random

tf.set_random_seed(777)  # reproducibility
learning_rate = 0.001
training_epochs = 15
batch_size = 100
num_models = 3

################
# Model 만들기 #
################
class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)

            # X, Y placeholder
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Conv Layer 1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            # Conv Layer 2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            # Conv Layer 3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            # FC Layer 1
            flat = tf.reshape(dropout3, [-1, 2048])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            # FC Layer 2
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        is_correct = tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_train, y_train, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_train, self.Y: y_train, self.training: training})

class Ensemble:
    def __init__(self, sess, models):
        self.sess = sess
        self.models = models

    def predict(self, x_i):
        prediction = np.zeros([1, 10])
        for m_idx, m in enumerate(self.models):
            p = m.predict(x_i)
            prediction += p
        total_prediction = tf.arg_max(prediction, 1)
        return self.sess.run(total_prediction)

    def get_accuracy(self, x_test, y_test):
        predictions = np.zeros([len(x_test), 10])
        for m_idx, m in enumerate(self.models):
            p = m.predict(x_test)
            predictions += p
        ensemble_is_correct = tf.equal(tf.arg_max(predictions, 1), tf.arg_max(y_test, 1))
        ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_is_correct, tf.float32))
        return self.sess.run(ensemble_accuracy)

    def print_each_accuracy(self, x_test, y_test):
        for m_idx, m in enumerate(self.models):
            print(m_idx, " Accuracy: ", m.get_accuracy(x_test, y_test))
        return

#######################
# MNIST Data 불러오기 #
#######################
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # size(60000, 10000)
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)  # y data는 one_hot 변환
x_train, x_test = tf.reshape(x_train, [-1, 784]), tf.reshape(x_test, [-1, 784])  # x data는 28x28 배열을 784x1로 변환
with tf.Session() as sess:
    x_train, x_test, y_train, y_test = sess.run([x_train, x_test, y_train, y_test])  # 위의 one_hot, reshape은 tensor이므로 session run을 통해 np.array로 다시 변환
x_train, x_test = x_train/255.0, x_test/255.0  # x는 0~255 범위의 RGB 값이므로 이를 normalize

##################
# Graph 실행하기 #
##################
with tf.Session() as sess:
    models = []
    for m in range(num_models):
        models.append(Model(sess, "model"+str(m)))

    sess.run(tf.global_variables_initializer())

    print("Learning Start!")
    for epoch in range(training_epochs):
        avg_cost_list = np.zeros(len(models))
        total_batch = int(len(x_train) / batch_size)

        for i in range(total_batch):
            x_batch = x_train[batch_size*i:batch_size*(i+1)]  # 처음부터 data를 np.array로 가져왔기 때문에
            y_batch = y_train[batch_size*i:batch_size*(i+1)]  # batch는 그냥 잘라서 쓰면 됨

            for m_idx, m in enumerate(models):
                c, _ = m.train(x_batch, y_batch)
                avg_cost_list[m_idx] += c / total_batch

        print("Epoch: {:4d}, cost: ".format(epoch+1), avg_cost_list)

    print("Learning Finished!")

    ensemble = Ensemble(sess, models)
    ensemble.print_each_accuracy(x_test, y_test)
    print("Ensemble Accuracy: ", ensemble.get_accuracy(x_test, y_test))

    ####################################################
    # 임의로 test data 하나를 출력하고 이를 prediction #
    ####################################################
    for i in range(100):
        r = random.randint(0, len(x_test) - 1)

        plt.imshow(x_test[r:r+1].reshape(28, 28)*255.0, cmap="Greys", interpolation="nearest")
        plt.show()

        print("아마 이 숫자는 \"{}\" 일 것 같아.".format(ensemble.predict(x_test[r:r+1])))
        input()



