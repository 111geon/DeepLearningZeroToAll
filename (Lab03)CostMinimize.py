import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import matplotlib.pyplot as plt

###[Data training set]###
x_data = [1,2,3]
y_data = [1,2,3]

###[Graph build]###
# W = tf1.Variable(tf1.random_normal([1]), name='weight')
W = tf1.Variable(5.0)
X = tf1.placeholder(tf1.float32)
Y = tf1.placeholder(tf1.float32)
# Our hypothesis for linear model X * W
hypothesis = X * W

###[cost/loss function]###
cost = tf1.reduce_mean(tf1.square(hypothesis - Y))

###[Minimize]###
learning_rate = 0.1
# gradient = tf1.reduce_mean((W * X - Y) * X)
gradient = tf1.reduce_mean((W * X - Y) * X * 2)
descent = W - learning_rate * gradient
update = W.assign(descent)  # node 는 그냥 '='로 assign이 불가하여 method 사용해야함.
# optimizer 함수를 이용하면 cost 함수 미분 필요 없이 자동으로 계산 가능하다.
optimizer = tf1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
# Get gradients
gvs = optimizer.compute_gradients(cost)
# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

###[Launch the graph in a session]###
sess = tf1.Session()
# Initializes global variables in the graph.
sess.run(tf1.global_variables_initializer())

###[Plotting cost function]###
# W_val = []
# cost_val = []
# for i in range(-30, 50):
#     feed_W = i * 0.1
#     curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
#     W_val.append(curr_W)
#     cost_val.append(curr_cost)
#
# # Show the cost function
# plt.plot(W_val, cost_val)
# plt.show()

###[Graph runing, print learning steps]###
# for step in range(21):
#     sess.run(update, feed_dict={X: x_data, Y: y_data})
#     print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

# for step in range(100):
#     print(step, sess.run(W))
#     sess.run(train, feed_dict={X: x_data, Y: y_data})

###[gvs 이용하여 gradient 계산 값 확인해보기]###
for step in range(100):
    print(step, sess.run([gradient, W, gvs], feed_dict={X: x_data, Y: y_data}))
    sess.run(apply_gradients, feed_dict={X: x_data, Y: y_data})

###########################
###[TensorFlow 2.0 기준]###
###########################
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd = tf.keras.optimizers.SGD(lr=0.1)
tf.model.compile(loss='mse', optimizer=sgd)

tf.model.summary()

# fit() trains the model and returns history of train
history = tf.model.fit(x_train, y_train, epochs=100)

y_predict = tf.model.predict(np.array([5,4]))
print(y_predict)

# Plot training % validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
