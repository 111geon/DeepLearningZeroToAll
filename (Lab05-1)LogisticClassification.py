import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf1
# tf1.disable_v2_behavior()
import tensorflow as tf

xy = np.array([[1,2,0],
               [2,3,0],
               [3,1,0],
               [4,3,1],
               [5,3,1],
               [6,2,1]])
x_data = xy[:,:-1]
y_data = xy[:,[-1]]

# X = tf1.placeholder(tf1.float32, shape=[None, 2])
# Y = tf1.placeholder(tf1.float32, shape=[None, 1])
# W = tf1.Variable(tf1.random_normal([2,1]), name="weight")
# b = tf1.Variable(tf1.random_normal([1]), name='bias')
#
# hypothesis = tf1.sigmoid(tf1.matmul(X, W) + b)
# # tf1.div(1., 1. + tf1.exp(tf1.matmul(X, W) + b))
#
# cost = -tf1.reduce_mean(Y*tf1.log(hypothesis) + (1-Y)*tf1.log(1-hypothesis))
#
# train = tf1.train.GradientDescentOptimizer(learning_rate=1e-02).minimize(cost)
#
# # Accuracy computation
# # True if hypothesis > 0.5 else False
# predicted = tf1.cast(hypothesis > 0.5, dtype = tf1.float32)
# accuracy = tf1.reduce_mean(tf1.cast(tf1.equal(predicted, Y), dtype=tf1.float32))
#
# with tf1.Session() as sess:
#     sess.run(tf1.global_variables_initializer())
#
#     for step in range(10001):
#         cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
#         if step % 200 == 0:
#             print(step, cost_val)
#
#     h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
#     print("\nHypothesis: \n", h, "\nCorrect (Y): \n", c, "\nAccuracy: \n", a)

###########################
###[TensorFlow 2.0 기준]###
###########################

xy = np.array([[1,2,0],
               [2,3,0],
               [3,1,0],
               [4,3,1],
               [5,3,1],
               [6,2,1]])
x_data = xy[:,:-1]
y_data = xy[:,[-1]]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))
tf.model.add(tf.keras.layers.Activation("sigmoid"))

tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=5000)
print("Accuracy: ", history.history['accuracy'][-1])

y_predict = tf.model.predict(np.array([[4.5, 3]]))
print("Predict: {0}".format(y_predict))

evaluate = tf.model.evaluate(x_data, y_data)
print("Loss: {0}, Accuracy: {1}".format(evaluate[0], evaluate[1]))
