########################
####[TensorFlow 1.0]####
########################
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)  # Noise term prevents zero division

xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

y_min = np.min(xy[:,[-1]], 0)
y_max = np.max(xy[:,[-1]], 0)
xy = MinMaxScaler(xy)

x_data = xy[:,:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, hyp_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
        if step % 10 == 0:
            print("Step: {:5}\tCost: {:.3f}\nPrediction: \n{}".format(step, cost_val, y_min+hyp_val*(y_max-y_min)))
