import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
tf.set_random_seed(777)

########################
# Better Data Creation #
########################
# sample data creation
sample = "if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and " \
         "work, but rather teach them to long for the endless immensity of the sea."
# len(sample) = 180
idx2chr = list(set(sample))  # [' ', 'i', 't', 'w', 'y', 'u', 'n', 'f', 'o', 'a']
chr2idx = {c: i for i, c in enumerate(idx2chr)}  # {' ': 0, 'i': 1, 't': 2, 'w': 3, 'y': 4, 'u': 5, 'n': 6, 'f': 7, 'o': 8, 'a': 9}

# Hyper Parameters
input_dim = len(chr2idx)  # RNN input size  # 25
hidden_size = len(chr2idx)  # RNN output size
num_classes = len(chr2idx)  # final output size (RNN or softmax, etc.)
sequence_length = 10  # number of LSTM rollings
learning_rate = 0.1

# x_data, y_data creation
x_data = []
y_data = []
for i in range(len(sample)-sequence_length):
    x_str = sample[i:i+sequence_length]
    y_str = sample[i+1:i+sequence_length+1]
    x = [chr2idx[c] for c in x_str]
    y = [chr2idx[c] for c in y_str]
    x_data.append(x)
    y_data.append(y)
# [[8, 0, 12, 4, 9, 22, 12, 19, 23, 2],
#                 ...
#  [0, 12, 4, 9, 22, 12, 19, 23, 2, 1]]
batch_size = len(x_data)  # 170

# x_one_hot, y_one_hot creation
x_one_hot = tf.one_hot(x_data, input_dim)
y_one_hot = tf.one_hot(y_data, input_dim)
with tf.Session() as sess:
    x_one_hot, y_one_hot = sess.run([x_one_hot, y_one_hot])  # (?, sequence_length, input_dim)
# [[[0. 0. 0. ... 0. 0. 0.]
#   [1. 0. 0. ... 0. 0. 0.]
#   [0. 0. 0. ... 0. 0. 0.]
#   ...
#   [0. 0. 0. ... 0. 0. 0.]
#   [0. 0. 0. ... 0. 1. 0.]
#   [0. 0. 1. ... 0. 0. 0.]]
#  ...
#  [[1. 0. 0. ... 0. 0. 0.]
#   [0. 0. 0. ... 0. 0. 0.]
#   [0. 0. 0. ... 0. 0. 0.]
#   ...
#   [0. 0. 0. ... 0. 1. 0.]
#   [0. 0. 1. ... 0. 0. 0.]
#   [0. 1. 0. ... 0. 0. 0.]]

###############
# graph build #
###############
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.float32, [None, sequence_length, input_dim])

cell = tf.keras.layers.LSTMCell(units=hidden_size, input_shape=(sequence_length, input_dim))
cells = tf.keras.layers.StackedRNNCells([cell]*2)
rnn = tf.keras.layers.RNN(cells, return_sequences=True)
cell_outputs = rnn(X)  # (170, 10, 25)
cell_outputs = tf.reshape(cell_outputs, [-1, hidden_size])  # (1700, 25)

W1 = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=[hidden_size, num_classes]))
b1 = tf.Variable(tf.random_normal([num_classes]))
hypothesis = tf.matmul(cell_outputs, W1) + b1
outputs = tf.reshape(hypothesis, [batch_size, sequence_length, num_classes])  # (170, 10, 25)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=Y)
# sequence_loss와 달리 Y에 one_hot이 들어가야했다.
# weight가 들어가는 cross_entropy인 sequence_loss는 tf.compat.v1.nn.weighted_cross_entropy_with_logits으로 있고 tensorflow2.0의 경우 model 안에 포함되어 있다.
cost = tf.reduce_mean(cost_i)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predictions = tf.argmax(outputs, axis=2)  # (170, 10)

#############
# Graph Run #
#############
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(300):
        c, _ = sess.run([cost, train], feed_dict={X: x_one_hot, Y: y_one_hot})
        results = sess.run(predictions, feed_dict={X: x_one_hot})
        print(i, "Cost: ", c, "Prediction: ", results, "Answer: ", y_data)

        result_str = [idx2chr[c] for c in np.squeeze(results)[0]]
        print("Prediction String: ", "".join(result_str))

    for j, result in enumerate(results):
        if j == 0:
            print(''.join([idx2chr[t] for t in result]), end='')
        else:
            print(idx2chr[result[-1]], end='')



