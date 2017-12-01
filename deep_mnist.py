import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

n_layers = 3  # no of layers (excluding the output layer)
n_nodes_list = [500, 500, 500]  # no of nodes in each layer (excluding the output layer)
n_classes = 10  # 10 classes for 0-9 digits
batch_size = 100
learning_rate = 0.01
training_iteration = 5

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])


# returns tensorflow variable of desired shape
def get_variable(shape):
    return tf.Variable(tf.random_normal(shape))


# building computation graph for model
def neural_network_model(data):
    n_nodes_prev_layer = 784
    prev_layer = data
    for i in range(n_layers):
        weights = get_variable([n_nodes_prev_layer, n_nodes_list[i]])
        biases = get_variable([n_nodes_list[i]])
        layer = tf.matmul(prev_layer, weights) + biases
        layer = tf.nn.relu(layer)
        n_nodes_prev_layer = n_nodes_list[i]
        prev_layer = layer

    # weights and biases for output layer
    weights = get_variable([n_nodes_list[len(n_nodes_list) - 1], n_classes])
    biases = get_variable([n_classes])
    layer = tf.matmul(prev_layer, weights)+biases
    return layer


# to train the model
def train_neural_network(dataset):
    model = neural_network_model(dataset)
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(training_iteration):
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)

            for _ in range(total_batch):
                x_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})
                avg_cost += sess.run(cost_function, feed_dict={x: x_batch, y: y_batch})

            print('Iteration no ', i, ' complete')
            predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
            print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print('Training completed')


train_neural_network(x)
