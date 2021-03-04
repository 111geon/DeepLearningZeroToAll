import numpy as np
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
tf1.set_random_seed(777)

# xy = np.loadtxt("(Lab04-2)LoadingDataFromFile.csv", delimiter=',', dtype=np.float32)
#
# x_data = xy[:, :-1]
# y_data = xy[:, [-1]]
#
# # Make sure the shape and data are OK
# print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data, len(y_data))
# print(type(x_data), type(y_data))
#
# # Placeholders for a tensor that will be always fed.
# X = tf1.placeholder(tf1.float32, shape=[None, 3])
# Y = tf1.placeholder(tf1.float32, shape=[None, 1])
#
# W = tf1.Variable(tf1.random_normal([3, 1]), name="weight")
# b = tf1.Variable(tf1.random_normal([1]), name="bias")
#
# # Hypothesis
# hypothesis = tf1.matmul(X, W) + b
#
# # Simplified cost/loss function
# cost = tf1.reduce_mean(tf1.square(hypothesis - Y))
#
# # Minimize
# optimizer = tf1.train.GradientDescentOptimizer(learning_rate = 1e-5)
# train = optimizer.minimize(cost)
#
# # Launch the graph in a session
# sess = tf1.Session()
# sess.run(tf1.global_variables_initializer())
#
# # Set up feed_dict variables inside the loop
# for step in range(2001):
#     cost_val, hyp_val, _ = sess.run([cost, hypothesis, train], feed_dict = {X: x_data, Y: y_data})
#     if step % 10 == 0:
#         print(step, "Cost: ", cost_val, "\nPrediction: ", hyp_val)
#
# # Ask my score
# print("Your score will be ", sess.run(hypothesis, feed_dict = {X: [[100, 70, 101]]}))
# print("Other scores will be ", sess.run(hypothesis, feed_dict = {X: [[60, 70, 110], [90, 100, 80]]}))


##################################################
###[Queue Runners_Batch를 이용한 Data handling]###
##################################################
# input에 사용되는 csv파일이 다수이고 training data의 크기가 허용 메모리를 초과할 때 data를 다루는 방법

# Filename Queue 생성
filename_queue = tf1.train.string_input_producer(["(Lab04-2)LoadingDataFromFile.csv"], shuffle = False, name = "filename_queue")

# Filename Queue에 사용할 reader 생성
reader = tf1.TextLineReader()
key, value = reader.read(filename_queue)

# reader에 적용할 decoder 생성
record_defaults = [[0.], [0.], [0.], [0.]]  # [[0.] for row in range(10000)] 로 row 갯수에 큰수도 적용 가능.
xy = tf1.decode_csv(value, record_defaults = record_defaults)

# Collect batches in csv
train_x_batch, train_y_batch = tf1.train.batch([xy[0:-1], xy[-1:]], batch_size=10)  # batch_size로 행의 갯수가 이미 정해졌기 때문에 열만 인덱싱

# Placeholders for a tensor that will be always fed.
X = tf1.placeholder(tf1.float32, shape=[None, 3])
Y = tf1.placeholder(tf1.float32, shape=[None, 1])

W = tf1.Variable(tf1.random_normal([3, 1]), name="weight")
b = tf1.Variable(tf1.random_normal([1]), name="bias")

# Hypothesis
hypothesis = tf1.matmul(X, W) + b

# Simplified cost/loss function
cost = tf1.reduce_mean(tf1.square(hypothesis - Y))

# Minimize
optimizer = tf1.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf1.Session()
sess.run(tf1.global_variables_initializer())

# Start populating the filename queue. (Queue 관리)
coord = tf1.train.Coordinator()
threads = tf1.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_batch, Y:y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction: \n", hy_val)

coord.request_stop()
coord.join(threads)
