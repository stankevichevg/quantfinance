import tensorflow as tf
import numpy as np

class Actor:
    """
    Актор. Апроксиматор функции политики.
    """
    def __init__(self, input_size, action_space_size, learning_rate=0.001, scope="Actor"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [input_size, input_size], "state")
            self.training = tf.placeholder(tf.bool, name="training")
            self.action = tf.placeholder(tf.int32, name="action")
            self.td_error = tf.placeholder(tf.float32, name="td_error")
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
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
            # Дропаут слой для предотвращения переобучения
            self.dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=self.training)
            # Выходной слой - вероятности действий
            self.actions_probabilities = tf.layers.dense(
                inputs=self.dropout,
                units=action_space_size,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.softmax
            )
            # Прибавим малое положительное чтобы не было проблем с отрицательными значениями
            log_prob = tf.log(self.actions_probabilities[0, self.action] + 1e-05)
            self.loss = -tf.reduce_mean(log_prob * self.td_error)

            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def choose_action(self, state, sess=None):
        sess = sess or tf.get_default_session()
        actions_probabilities, dropout = sess.run(
            [self.actions_probabilities, self.dropout],
            {
                self.state: state,
                self.training: False
            }
        )
        return np.random.choice(np.arange(actions_probabilities.shape[1]), p=actions_probabilities.ravel())

    def update(self, state, td_error, action, sess=None):
        sess = sess or tf.get_default_session()
        _, loss, actions_probabilities = sess.run(
            [self.train_op, self.loss, self.actions_probabilities],
            {
                self.state: state,
                self.action: action,
                self.training: True,
                self.td_error: td_error
            }
        )
        return loss, actions_probabilities
