import tensorflow as tf
# import tensorflow.compat.v1 as tf1
# tf1.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

# ###########################
# ###[TensorFlow 1.0 기준]###
# ###########################
# ###[Train data set]###
# # x1_data = [73.,93.,89.,96.,73.]
# # x2_data = [80.,88.,91.,98.,66.]
# # x3_data = [75.,93.,90.,100.,70.]
# x_data = [[73.,80.,75.],
#           [93.,88.,93.],
#           [89.,91.,90.],
#           [96.,98.,100.],
#           [73.,66.,70.]]
# # y_data = [152.,185.,180.,196.,142.]
# y_data = [[152.],
#           [185.],
#           [180.],
#           [196.],
#           [142.]]
#
# ###[Placeholders for a tensor that will be always fed]###
# # x1 = tf1.placeholder(tf1.float32)
# # x2 = tf1.placeholder(tf1.float32)
# # x3 = tf1.placeholder(tf1.float32)
# X = tf1.placeholder(tf1.float32, shape=[None, 3])
#
# # Y = tf1.placeholder(tf1.float32)
# Y = tf1.placeholder(tf1.float32, shape=[None, 1])
#
# # w1 = tf1.Variable(tf1.random_normal([1]), name='weight1')
# # w2 = tf1.Variable(tf1.random_normal([1]), name='weight2')
# # w3 = tf1.Variable(tf1.random_normal([1]), name='weight3')
# W = tf1.Variable(tf1.random_normal([3, 1]), name='weight')
#
# b = tf1.Variable(tf1.random_normal([1]), name='bias')
#
# # hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
# hypothesis = tf.matmul(X, W) + b
#
# ###[Cost/Loss function]###
# cost = tf1.reduce_mean(tf1.square(hypothesis - Y))
# # Minimize
# optimizer = tf1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(cost)
#
# ###[Launch the graph in a session]###
# sess = tf1.Session()
# # Initialize global variables in the graph
# sess.run(tf1.global_variables_initializer())
#
# ###[Graph run]###
# for step in range(2001):
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
#         # feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
#         feed_dict = {X: x_data, Y: y_data})
#     if step % 10 == 0:
#         print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)

###########################
###[TensorFlow 2.0 기준]###
###########################
# x_data = np.array([[73.,80.,75.],
#           [93.,88.,93.],
#           [89.,91.,90.],
#           [96.,98.,100.],
#           [73.,66.,70.]])
# y_data = np.array([[152.],
#           [185.],
#           [180.],
#           [196.],
#           [142.]])
#
# tf.model = tf.keras.Sequential()
#
# tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))  # input_dim=3 gives multi-variable regression
# tf.model.add(tf.keras.layers.Activation('linear'))  # this line can be omitted, as linear activation is default
# # advanced reading https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
#
# tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
# tf.model.summary()
# history = tf.model.fit(x_data, y_data, epochs=100)
#
# y_predict = tf.model.predict(np.array([[72.,93.,90.]]))
# print(y_predict)

###################################
###[TensorFlow 2.0 좀 더 자세히]###
###################################
# data matrix        x1  x2  x3   y
data = np.array([[73.,80.,75.,152.],
                    [93.,88.,93.,185.],
                    [89.,91.,90.,180.],
                    [96.,98.,100.,196.],
                    [73.,66.,70.,142.]], dtype = np.float32)

X_data = data[:,:-1]  # 5x3 matrix
Y_data = data[:,[-1]]  # 5x1 matrix

W = tf.Variable(tf.random.normal([3,1]))  # 3x1 matrix
b = tf.Variable(tf.random.normal([1]))  # scalar

learning_rate = 0.00001

# Hypothesis function
@tf.function
def predict(X):
    return tf.matmul(X, W) + b

print("    i |       cost")
for i in range(2000+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X_data) - Y_data)))

    # W와 b의 기울기를 계산
    W_grad, b_grad = tape.gradient(cost, [W, b])

    # W와 b를 업데이트
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))

print("W: ")
print(W.read_value().numpy())
print("b: ")
print(b.read_value().numpy())
