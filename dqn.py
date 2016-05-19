#!/usr/bin/env python2

import tensorflow as tf
from solitaire import Solitaire

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def createNet():
    Wconv1 = weight([10, 10, 3, 32])
    bconv1 = bias([32])

    Wconv2 = weight([7, 7, 32, 64])
    bconv2 = bias([64])

    Wconv3 = weight([4, 4, 64, 64])
    bconv3 = bias([64])

    Wfc1 = weight([3136, 512])
    bfc1 = bias([512])

    Wfc2 = weight([512, 5])
    bfc2 = bias([5])


    x = tf.placeholder("float", [None, 250, 250, 3])

    hconv1 = tf.nn.relu(tf.nn.conv2d(x, Wconv1, [1, 5, 5, 1], "SAME") + bconv1)

    hconv2 = tf.nn.relu(tf.nn.conv2d(hconv1, Wconv2, [1, 5, 5, 1], "SAME") + bconv2)

    hconv3 = tf.nn.relu(tf.nn.conv2d(hconv2, Wconv3, [1, 2, 2, 1], "SAME") + bconv3)

    hfc1 = tf.nn.relu(tf.matmul(tf.reshape(hconv3, [-1, 3136]), Wfc1) + bfc1)

    output = tf.matmul(hfc1, Wfc2) + bfc2

    return x, output


def train(x, output):
    sol = Solitaire()


def main():
    x, output = createNet()
    train(x, output)

if __name__ == "__main__":
    main()
