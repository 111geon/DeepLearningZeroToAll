import numpy as np

xy = np.array([[1,2,1,1,0,0,1],
               [2,1,3,2,0,0,1],
               [3,1,3,4,0,0,1],
               [4,1,5,5,0,1,0],
               [1,7,5,5,0,1,0],
               [1,2,5,6,0,1,0],
               [1,6,6,6,1,0,0],
               [1,7,7,7,1,0,0]])
x_data = xy[:,:4]
y_data = xy[:,4:]

############################
####[Tensorflow 1.0기준]####
############################
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# X = tf.placeholder("float", [None, 4])
# Y = tf.placeholder("float", [None, 3])
# nb_classes = 3
#
# W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
# b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
#
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
#
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))  # axis는 어느 dim에서의 합을 구할 것인지에 대한 것
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(2000+1):
#         sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
#         if step % 200 == 0:
#             print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
#
#     all = sess.run(hypothesis, feed_dict={X: [[1,11,7,9],
#                                               [1,3,4,3],
#                                               [1,1,0,1]]})
#     print(all, sess.run(tf.arg_max(all, 1)))

############################
####[Tensorflow 2.0기준]####
############################
import tensorflow as tf

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(input_dim=4, units=3, use_bias=True))

tf.model.add(tf.keras.layers.Activation("softmax"))

tf.model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=["accuracy"])
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=2000)

print("------------------")
a = tf.model.predict(np.array([[1,11,7,9]]))
print(a, tf.keras.backend.eval(tf.argmax(a, axis=1)))



