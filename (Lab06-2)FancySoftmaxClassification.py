####################
##[Tensorflow 1.0]##
####################
# import numpy as np
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# xy = np.loadtxt("(Lab06-2)data-04-zoo.csv", delimiter=",", dtype=np.float32)
# x_data = xy[:, :-1]
# y_data = xy[:, [-1]]
#
# X = tf.placeholder(tf.float32, [None, 16])
# Y = tf.placeholder(tf.int32, [None, 1])
#
# Y_onehot = tf.one_hot(Y, 7)  # classes 갯수: 7  # one_hot은 [3] -> [0,0,0,1,0,0,0]
# Y_onehot = tf.reshape(Y_onehot, [-1, 7])  # tf.one_hot은 shape을 [?, 1, 7]로 만들어 버리기 때문에 차원 제거가 필요하다.
# # 1) reshape은 원하는 shape를 직접 입력하여 바꿀 수 있다.
# #    특히 shape에 -1를 입력하면 고정된 차원은 우선 채우고 남은 부분을 알아서 채워준다.
# # 2) squeeze는 차원 중 사이즈가 1인 것을 찾아 스칼라값으로 바꿔 해당 차원을 제거한다.
# # 3) expand_dims는 axis로 지정된 차원을 추가한다.
#
# W = tf.Variable(tf.random_normal([16, 7]), name="weight")
# b = tf.Variable(tf.random_normal([7]), name="bias")
#
# logits = tf.matmul(X, W) + b
# hypothesis = tf.nn.softmax(logits)
# cost_i = tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= Y_onehot)
# cost = tf.reduce_mean(cost_i)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(cost)
#
# prediction = tf.argmax(hypothesis, 1)  # argmax는 [0.1, 0.9, 0] -> [1]
# correct_prediction = tf.equal(prediction, tf.argmax(Y_onehot, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(2001):
#         sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
#         if step % 100 == 0:
#             loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
#             print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
#
#     pred = sess.run(prediction, feed_dict={X: x_data})
#     for p, y in zip(pred, y_data.flatten()):  # flatten()은 [[0],[1]] -> [0, 1]
#         print("[{}] Prediction: {} True Y: {}".format(p==int(y), p, int(y)))

####################
##[Tensorflow 2.0]##
####################
import numpy as np
import tensorflow as tf
xy = np.loadtxt("(Lab06-2)data-04-zoo.csv", delimiter=",", dtype=np.float32)
x_data = xy[:,:-1]
y_data = xy[:, [-1]]
print(xy)
print(x_data)
print(y_data)

y_onehot = tf.keras.utils.to_categorical(y_data, 7)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=7, input_dim=16, activation="softmax"))
tf.model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=["accuracy"])
tf.model.summary()

tf.model.fit(x_data, y_onehot, epochs=1000)

test_data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]])
pred = tf.model.predict(test_data)
print(pred, tf.keras.backend.eval(tf.argmax(pred, 1)))

pred_all = tf.model.predict(x_data)
pred_all = tf.keras.backend.eval(tf.argmax(pred_all, 1))
for p, y in zip(pred_all, y_data.flatten()):
    print("{}, {}, {}".format(p==int(y), p, int(y)))

