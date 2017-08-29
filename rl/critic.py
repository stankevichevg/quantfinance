import tensorflow as tf
import numpy as np


class Critic:
    """
    Критик. Апроксиматор функции ценности состояния.
    """
    def __init__(self, input_size, action_space_size, learning_rate=0.01, scope="Critic"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [input_size, input_size], "state")
            self.training = tf.placeholder(tf.bool, name="training")
            self.td_error = tf.placeholder(tf.float32, name="td_error")
            self.value = tf.placeholder(tf.float32, name="value")
            self.td_target = tf.placeholder(tf.float32, name="td_target")
            # Входной слой
            input_layer = tf.reshape(self.state, [-1, input_size, input_size, 1])
            # Первый сверточный слой
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu
            )
            # Первый пулинг слой
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            # Второй сверточный слой
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=action_space_size,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu
            )
            # Второй пулинг слой
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            # Полносвязный слой
            pool2_flat = tf.reshape(pool2, [-1, int(input_size / 4) * int(input_size / 4) * action_space_size])
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, kernel_initializer=tf.zeros_initializer())
            # Дропаут слой для предотвращения переобучения
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=self.training)
            # Выходной слой - ценность состояния. Линейная комбинация предыдущего слоя.
            self.value = tf.layers.dense(
                inputs=dropout,
                units=1,
                activation=None
            )
            self.loss = tf.squared_difference(self.value, self.td_target)
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(
            self.value,
            {
                self.state: state,
                self.training: False
            }
        )

    def update(self, state, td_target, sess=None):
        sess = sess or tf.get_default_session()
        _, loss = sess.run(
            [self.train_op, self.loss],
            {
                self.state: state,
                self.td_target: td_target,
                self.training: True
            }
        )
        return loss
