import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import random


#######################
# MNIST Data 불러오기 #
#######################
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # size(60000, 10000)
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)  # y data는 one_hot 변환
x_train, x_test = tf.reshape(x_train, [-1, 784]), tf.reshape(x_test, [-1, 784])  # x data는 28x28 배열을 784x1로 변환
with tf.Session() as sess:
    x_train, x_test, y_train, y_test = sess.run([x_train, x_test, y_train, y_test])  # 위의 one_hot, reshape은 tensor이므로 session run을 통해 np.array로 다시 변환
x_train, x_test = x_train/255.0, x_test/255.0  # x는 0~255 범위의 RGB 값이므로 이를 normalize

keep_prob = tf.placeholder(tf.float32)

################
# Conv Layer 1 #
################
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28 x 28 x 1 (black/white)  # -1은 input이 몇개 들어가던 자동으로 하도록
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape = (?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # 3x3 size 32 filters with black/white
# Conv -> (?, 28, 28, 32)
# Pool -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")  # strides 1 x 1  # padding으로 사이즈 유지
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # kernal size 2 x 2. filter 크기
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

################
# Conv Layer 2 #
################
# L2 ImgIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# Conv -> (?, 14, 14, 64)
# Pool -> (?,  7,  7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

################
# Conv Layer 3 #
################
# L3 ImgIn shape = (?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
# Conv -> (?, 7, 7, 128)
# Pool -> (?, 4, 4, 128)
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding="SAME")
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
L3 = tf.reshape(L3, [-1, 4*4*128])  # FCL에 넣기 전 입체 모양의 tensor를 펴준다.
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

###########################
# Fully Connected Layer 1 #
###########################
# L4 ImgIn shape = (?, 2048)
W4 = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=[2048, 625]))
b4 = tf.Variable(tf.random_normal([625]), name="bias4")
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

###########################
# Fully Connected Layer 2 #
###########################
# L5 ImgIn shape = (?, 625)
W5 = tf.Variable(tf.keras.initializers.glorot_uniform()(shape=[625, 10]))
b5 = tf.Variable(tf.random_normal([10]), name="bias4")
hypothesis = tf.matmul(L4, W5) + b5


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



