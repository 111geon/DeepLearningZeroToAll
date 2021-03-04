import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

###[training data 정의]###
# 고정된 training set을 이용하여 train
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# Placeholders를 이용하여 train
X = tf1.placeholder(tf1.float32, shape=[None])
Y = tf1.placeholder(tf1.float32, shape=[None])

###[구하려는 W, b 임의 설정]###
# tf.Variable은 tensorflow가 학습시키는 과정에서 변하는 trainable한 값이다.
# tf.random_normal(Shape)
W = tf1.Variable(tf1.random_normal([1]), name='weight')
b = tf1.Variable(tf1.random_normal([1]), name='bias')

###[Our hypothesis XW+b]###
# hypothesis = x_train * W + b

# using placeholder
hypothesis = X * W + b

###[Cost/loss function]###
# cost = tf1.reduce_mean(tf1.square(hypothesis - y_train))

# using placeholder
cost = tf1.reduce_mean(tf1.square(hypothesis - Y))

###[Cost function minimize]###
optimizer = tf1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf1.Session()
sess.run(tf1.global_variables_initializer())

###[Session Run]###
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

# using placeholder
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
        feed_dict = {X: [1,2,3,4,5], Y: [2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

###[Testing our model]###
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

###########################
###[TensorFlow 2.0 기준]###
###########################
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

tf.model = tf.keras.Sequential()
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig(y'-y)^2

# prints summart of the model to the terminal
tf.model.summary()

# fir() executes training
tf.model.fit(x_train, y_train, epochs=200)

# predict() returns predicted value
y_predict = tf.model.predict(np.array([5,4]))
print(y_predict)
