import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
tf.set_random_seed(777)
import matplotlib.pyplot as plt

#################
# Data Creation #
#################
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)  # Noise term prevents zero division

# Hyper Parameters
seq_length = 7
data_dim = 5
hidden_dim = 5
output_dim = 1
learning_rate = 0.01
iteration = 500

xy = np.loadtxt("(Lab12-4)stock.csv", delimiter=',')
xy = xy[::-1]  # 역순으로  # 시간 순으로 정렬함을 위함.
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]

x_data = []  # (725, 7, 5)
y_data = []  # (725, 1)
for i in range(len(y) - seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]
    x_data.append(_x)
    y_data.append(_y)

train_size = int(len(y_data)*0.7)
test_size = len(y_data) - train_size
x_train, x_test = np.array(x_data[0:train_size]), np.array(x_data[train_size:len(x_data)])  # (507, 7, 5), (218, 7, 5)
y_train, y_test = np.array(y_data[0:train_size]), np.array(y_data[train_size:len(y_data)])  # (507, 1), (218, 1)

###############
# graph build #
###############
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

cell = tf.keras.layers.LSTMCell(units=hidden_dim, input_shape=(seq_length, data_dim))
cells = tf.keras.layers.StackedRNNCells([cell]*2)
rnn = tf.keras.layers.RNN(cells, return_sequences=True)
cell_outputs = rnn(X)  # (507, 7, 5)
cell_outputs = cell_outputs[:, -1]  # (507, 5)

W1 = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=[hidden_dim, output_dim]))
b1 = tf.Variable(tf.random_normal([output_dim]))
hypothesis = tf.matmul(cell_outputs, W1) + b1

cost = tf.reduce_sum(tf.square(hypothesis - Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        c, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
        print(i, c)

    prediction = sess.run(hypothesis, feed_dict={X: x_test})

plt.plot(y_test, label='Real')
plt.plot(prediction, label="Prediciton")
plt.show()

# 결과를 보면 겉으로는 패턴을 비슷하게 예측하는 것으로 보이나 확대해보면
# Prediction은 Real 보다 x축 방향으로 한칸 뒤늦게 나오는 것을 볼 수 있다.
# 이는 이전 Real 값을 흉내만 내고 있다는 의미이다.
# 이와같이 직전 패턴을 기반으로 전날 종가를 흉내만 내는 문제를 해결하기 위해서는 Bidirectional RNN을 사용해야한다.
# Bidirectional RNN 설명 참고: http://solarisailab.com/archives/1515
