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

with tf.name_scope("layer1") as scope:  # Add scope for better graph hierarchy
    W1 = tf.Variable(tf.random_normal([784, 20]), name="weight1")
    b1 = tf.Variable(tf.random_normal([20]), name="bias1")
    Layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

    W1_hist = tf.summary.histogram("weights1", W1)  # Tensorboard1: decide which tensors you want to log
    b1_hist = tf.summary.histogram("biases1", b1)  # Tensorboard1: decide which tensors you want to log
    Layer1_hist = tf.summary.histogram("Layer1", Layer1)  # Tensorboard1: decide which tensors you want to log

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([20, 10]))
    b3 = tf.Variable(tf.random_normal([10]))
    hypothesis = tf.nn.softmax(tf.matmul(Layer1, W3) + b3)

    W3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("biases3", b3)
    hypothesis_hist = tf.summary.histogram("Layer3", hypothesis)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost_summ = tf.summary.scalar("cost", cost)  # Tensorboard1: decide which tensors you want to log
summary = tf.summary.merge_all()  # Tensorboard2: Merge summaries

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


##################
# Graph 실행하기 #
##################
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Tensorboard3: Create summary writer
    writer = tf.summary.FileWriter("./logs/MNIST_logs_01")
    writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(x_train) / batch_size)

        for i in range(total_batch):
            x_batch = x_train[batch_size*i:batch_size*(i+1)]  # 처음부터 data를 np.array로 가져왔기 때문에
            y_batch = y_train[batch_size*i:batch_size*(i+1)]  # batch는 그냥 잘라서 쓰면 됨
            c, _, s = sess.run([cost, optimizer, summary], feed_dict={X:x_batch, Y:y_batch})  # Tensorboard3: summary도 run
            avg_cost += c / total_batch

        writer.add_summary(s, global_step=epoch)  # Tensorboard3: write summary
        print("Epoch: {:4d}, cost: {:.9f}".format(epoch+1, avg_cost))

    writer = tf.summary.FileWriter("./logs/MNIST_logs_01")  # Tensorboard4: summary 생성
    # Tensorboard5: [summary 확인하는 법]
    # Anaconda 가상환경에서 python을 작동 중이기 때문에 Anaconda Prompt에서 버전 맞춰준 후에 작업
    # summary 파일이 저장된 경로로 이동후 >tensorboard --logdir=. (첫 시도에 작동이 안되서 tensorboard를 uninstall 후 다시 install 했음)
    # 출력된 주소 크롬 탭에서 열기
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X:x_test, Y:y_test}))  # Tensor 하나의 출력은 sess.run 할 필요 없이 eval을 쓰면 편하다.

    ####################################################
    # 임의로 test data 하나를 출력하고 이를 prediction #
    ####################################################
    for i in range(100):
        r = random.randint(0, len(x_test) - 1)

        plt.imshow(x_test[r:r+1].reshape(28, 28)*255.0, cmap="Greys", interpolation="nearest")
        plt.show()

        print("아마 이 숫자는 \"{}\" 일 것 같아.".format(sess.run(tf.arg_max(hypothesis, 1), feed_dict={X:x_test[r:r+1]})[0]))
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
