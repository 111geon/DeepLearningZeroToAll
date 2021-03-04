import tensorflow as tf
import numpy as np
import os
# Info 로그를 필터링하려면 1, Warning 로그는 2, Error 로그는 3
# os.environ['TF_CPP_MIN_LOG_LEVEL']='1'


# [Tensorflow 낮은 버전 이용하고 싶을 때]
# import tensorflow.compat.v1 as tf1
# tf1.disable_v2_behavior()


# [tensorflow 버전확인]
# print(tf.__version__)


# [Hello World! 프로그램]
# hello = tf.constant("Hello, TensorFlow!")
# tf.print(hello)


# [노드 기반의 Computational Graph / 두개의 노드 더하기]
# Version 1 에서는 Session을 생성하여 Run 해야했지만 Version 2 부터는 Session 없이 메소드 이용

# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)
# node3 = tf.add(node1, node2)
# tf.print(node3)

# tf.add: 덧셈
# tf.subtract: 뺄셈
# tf.multiply: 곱셈
# tf.divide: 나눗셈
# tf.pow: n-제곱 (ex. tf.pow(2,3))
# tf.negative: 음수 부호
# tf.abs: 절대값
# tf.sign: 부호
# tf.round: 반올림
# tf.math.ceil: 올림
# tf.floor: 내림
# tf.math.square: 제곱
# tf.math.sqrt: 제곱근
# tf.maximum: 두 텐서의 각 원소에서 최댓값만 반환 (ex. tf.maximum(2, 3))
# tf.minimum: 두 텐서의 각 원소에서 최솟값만 반환
# tf.cumsum: 누적합 (ex. x = tf.constant([2,4,6,8]); print(tf.cumsum(x)))
# tf.math.cumprod: 누적곱


# [노드 기반의 Computational Graph / 그래프 만들기]
# Version 1 에서는 placeholder를 이용했지만 Version 2 부터는 tf.function를 이용하여 그래프를 생성

@tf.function
def adder(a, b):
    return a + b

A = tf.constant(1)
B = tf.constant(2)
print(adder(A, B))

C = tf.constant([1, 2])
D = tf.constant([3, 4])
print(adder(C, D))

E = tf.constant([[1, 2, 3], [4, 5, 6]])
F = tf.constant([[2, 3, 4], [5, 6, 7]])
print(adder(E, F))
