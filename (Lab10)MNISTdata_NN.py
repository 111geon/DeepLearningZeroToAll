import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import random

#######################
# MNIST Data 불러오기 #
#######################
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 60000개 data
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)  # y data는 one_hot 변환
x_train, x_test = tf.reshape(x_train, [-1, 784]), tf.reshape(x_test, [-1, 784])  # x data는 28x28 배열을 784x1로 변환
with tf.Session() as sess:
    x_train, x_test, y_train, y_test = sess.run([x_train, x_test, y_train, y_test])  # 위의 one_hot, reshape은 tensor이므로 session run을 통해 np.array로 다시 변환
x_train, x_test = x_train/255.0, x_test/255.0  # x는 0~255 범위의 RGB 값이므로 이를 normalize


################
# Graph 만들기 #
################
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("layer1") as scope:  # Add scope for better graph hierarchy
    W1 = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=[784, 256]))  # Xavier uniform initializer
    b1 = tf.Variable(tf.random_normal([256]), name="bias1")
    _Layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # ReLU
    Layer1 = tf.nn.dropout(_Layer1, keep_prob=keep_prob)  # Dropout node

with tf.name_scope("layer2") as scope:  # Neural Network
    W2 = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=[256, 256]))
    b2 = tf.Variable(tf.random_normal([256]), name="bias1")
    _Layer2 = tf.nn.relu(tf.matmul(Layer1, W2) + b2)
    Layer2 = tf.nn.dropout(_Layer2, keep_prob=keep_prob)

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=[256, 10]))
    b3 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.matmul(Layer2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)  # Adaptive Gradient Algorithm

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


##################
# Graph 실행하기 #
##################
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        for i in range(total_batch):
            x_batch = x_train[batch_size*i:batch_size*(i+1)]  # 처음부터 data를 np.array로 가져왔기 때문에
            y_batch = y_train[batch_size*i:batch_size*(i+1)]  # batch는 그냥 잘라서 쓰면 됨
            c, _ = sess.run([cost, optimizer], feed_dict={X:x_batch, Y:y_batch, keep_prob: 0.7})  # 학습시, dropout을 위한 keep_prob를 포함한 feed_dict
            avg_cost += c / total_batch

        print("Epoch: {:4d}, cost: {:.9f}".format(epoch+1, avg_cost))

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X:x_test, Y:y_test, keep_prob: 1}))  # 최종 예측 땐 dropout을 하지 않는다.

    ####################################################
    # 임의로 test data 하나를 출력하고 이를 prediction #
    ####################################################
    for i in range(100):
        r = random.randint(0, len(x_test) - 1)

        plt.imshow(x_test[r:r+1].reshape(28, 28)*255.0, cmap="Greys", interpolation="nearest")
        plt.show()

        print("아마 이 숫자는 \"{}\" 일 것 같아.".format(sess.run(tf.arg_max(hypothesis, 1), feed_dict={X:x_test[r:r+1], keep_prob: 1})[0]))
        input()


#######################
# Tensorflow 2.0 참고 #
#######################
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28,28)), #28 by 28 mnist input flatten
#     tf.keras.layers.Dense(10,activation='softmax')
# ])
#
# model.compile(optimizer='SGD',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test, verbose=2)
