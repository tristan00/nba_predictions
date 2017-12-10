import tensorflow as tf
from datamanager import get_features
import time
import numpy as np

n_classes = 2
batch_size = 10
total_epochs = 1000
nodes_per_layer = 500

def neural_network_model(input_x, input_lenth):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([input_lenth, nodes_per_layer])),
                      'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                      'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                      'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                      'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer])),
                      'biases': tf.Variable(tf.random_normal([nodes_per_layer]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_per_layer, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    prob = tf.placeholder_with_default(1.0, shape=())
    l1 = tf.add(tf.matmul(input_x, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.leaky_relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.leaky_relu(l3)
    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.leaky_relu(l4)
    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.leaky_relu(l5)
    l5_dropout = tf.nn.dropout(l5, prob)
    output = tf.add(tf.matmul(l5_dropout, output_layer['weights']), output_layer['biases'])
    return output, prob

def model():
    train_x, train_y, test_x, test_y = get_features()
    x = tf.placeholder('float', [None, len(train_x[0])])
    y = tf.placeholder('float', [None, n_classes])
    prediction, prob = neural_network_model(x, len(train_x[0]))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(total_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, prob: .5})
                epoch_loss += c
                i += batch_size
            print('epoch: {0}, total epoch loss: {1}'.format(epoch, epoch_loss))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy_float = accuracy.eval(session = sess, feed_dict = {x:test_x, y:test_y, prob:1.0})
        print('accuracy:', accuracy_float)

if __name__ == '__main__':
    model()

