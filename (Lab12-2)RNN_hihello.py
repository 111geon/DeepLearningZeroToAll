import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

sess = tf.InteractiveSession()

idx2char = ['h', 'i', 'e', 'l', 'o']
input_dim = 5
sequence_length = 6
hidden_size = 5
batch_size = 1
num_classes = 5
learning_rate = 0.1

#################
# Data Creation #
#################
def make_data(str):
    data = []
    for i in str:
        data.append(idx2char.index(i))
    data = [data]
    return data

x_data = make_data("hihell")
x_one_hot = tf.one_hot(x_data, len(idx2char))
y_data = make_data("ihello")
y_one_hot = tf.one_hot(y_data, len(idx2char))
x_one_hot, y_one_hot = sess.run([x_one_hot, y_one_hot])  # Tensor에서 array로 형변환 하지 않으면 feed할 수 없다.

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
sess.run(tf.global_variables_initializer())
print(sess.run([hypothesis], feed_dict={X: x_one_hot}))
for i in range(200):
    c, _ = sess.run([cost, train], feed_dict={X: x_one_hot, Y: y_one_hot})
    result = sess.run([prediction], feed_dict={X: x_one_hot})
    print(i, "Cost: ", c, "Prediction: ", result, "Answer: ", y_data)

    result_str = [idx2char[c] for c in np.squeeze(result)]
    print("Prediction String: ", "".join(result_str))


