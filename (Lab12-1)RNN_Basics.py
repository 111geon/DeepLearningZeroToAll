import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

with tf.variable_scope("3_batches") as scope:
    hidden_size=2
    cell = tf.keras.layers.LSTMCell(units=hidden_size)
    print(cell.output_size, cell.state_size)

    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    print(x_data.shape)
    pp.pprint(x_data)
    
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
    outputs, final_memory_state, final_carry_state = rnn(x_data)

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    pp.pprint(final_memory_state.eval())
    pp.pprint(final_carry_state.eval())
