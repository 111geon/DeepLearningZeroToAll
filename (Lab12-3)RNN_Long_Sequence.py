import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
tf.set_random_seed(777)

########################
# Better Data Creation #
########################
# sample data creation
sample = "if you want you"
idx2chr = list(set(sample))  # [' ', 'i', 't', 'w', 'y', 'u', 'n', 'f', 'o', 'a']
chr2idx = {c: i for i, c in enumerate(idx2chr)}  # {' ': 0, 'i': 1, 't': 2, 'w': 3, 'y': 4, 'u': 5, 'n': 6, 'f': 7, 'o': 8, 'a': 9}
sample_idx = [chr2idx[c] for c in sample]  # [1, 7, 0, 4, 8, 5, 0, 3, 9, 6, 2, 0, 4, 8, 5]

# Hyper Parameters
input_dim = len(chr2idx)  # RNN input size # 10
hidden_size = len(chr2idx)  # RNN output size # 10
num_classes = len(chr2idx)  # final output size (RNN or softmax, etc.) # 10
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of LSTM rollings # 14
learning_rate = 0.1

# x_data, y_data creation
x_data = [sample_idx[:-1]]  # [[5, 2, 1, 3, 9, 6, 1, 7, 0, 4, 8, 1, 3, 9]]
y_data = [sample_idx[1:]]  # [[2, 1, 3, 9, 6, 1, 7, 0, 4, 8, 1, 3, 9, 6]]
x_one_hot = tf.one_hot(x_data, num_classes)
y_one_hot = tf.one_hot(y_data, num_classes)
with tf.Session() as sess:
    x_one_hot, y_one_hot = sess.run([x_one_hot, y_one_hot])  # (?, sequence_length, input_dim)

###############
# graph build #
###############
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.float32, [None, sequence_length, input_dim])

cell = tf.keras.layers.LSTMCell(units=hidden_size, input_shape=(sequence_length, input_dim))
rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
cell_outputs, _memory_state, _carry_state = rnn(X)
cell_outputs = tf.reshape(cell_outputs, [-1, hidden_size])

W1 = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=[hidden_size, num_classes]))
b1 = tf.Variable(tf.random_normal([num_classes]))
hypothesis = tf.matmul(cell_outputs, W1) + b1

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y)
# sequence_loss와 달리 Y에 one_hot이 들어가야했다.
# weight가 들어가는 cross_entropy인 sequence_loss는 tf.compat.v1.nn.weighted_cross_entropy_with_logits으로 있고 tensorflow2.0의 경우 model 안에 포함되어 있다.
cost = tf.reduce_mean(cost_i)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, axis=1)

#############
# Graph Run #
#############
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        c, _ = sess.run([cost, train], feed_dict={X: x_one_hot, Y: y_one_hot})
        result = sess.run([prediction], feed_dict={X: x_one_hot})
        print(i, "Cost: ", c, "Prediction: ", result, "Answer: ", y_data)

        result_str = [idx2chr[c] for c in np.squeeze(result)]
        print("Prediction String: ", "".join(result_str))



