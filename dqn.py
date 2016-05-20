#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import random
from solitaire import Solitaire
from collections import deque

ACTIONS = ['select', 'up', 'down', 'left', 'right']
REPLAY_MEMORY = 40000 # 250*250*3*40000 is almost 8GB
OBSERVE = 30000

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

    Wfc1 = weight([1600, 512])
    bfc1 = bias([512])

    Wfc2 = weight([512, 5])
    bfc2 = bias([5])


    x = tf.placeholder("float", [None, 250, 250, 3])

    hconv1 = tf.nn.relu(tf.nn.conv2d(x, Wconv1, [1, 5, 5, 1], "SAME") + bconv1)

    hconv2 = tf.nn.relu(tf.nn.conv2d(hconv1, Wconv2, [1, 5, 5, 1], "SAME") + bconv2)

    hconv3 = tf.nn.relu(tf.nn.conv2d(hconv2, Wconv3, [1, 2, 2, 1], "SAME") + bconv3)

    hfc1 = tf.nn.relu(tf.matmul(tf.reshape(hconv3, [-1, 1600]), Wfc1) + bfc1)

    output = tf.matmul(hfc1, Wfc2) + bfc2

    return x, output


def train(x, output):
    sol = Solitaire()
    sess = tf.InteractiveSession()

    action = tf.placeholder("float", [None, 5])
    y = tf.placeholder("float", [None])
    output_action = tf.reduce_sum(tf.mul(output, action), 1)
    cost = tf.reduce_mean(tf.square(y - output_action))
    optimizer = tf.train.RMSPropOptimizer(1e-6).minimize(cost)

    D = deque()

    x_t, r_t = sol.step()
    exploration_rate = 1.0

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    t = 0
    while True:
        action_t = np.zeros([5])
        output_t = None

        # Sometimes (according to the exploration_rate)
        # pick an entirely random action instead of the
        # best prediction.
        if random.random() <= exploration_rate:
            action_idx = random.randrange(5)
        else:
            output_t = output.eval({x: [x_t]})[0]
            action_idx = np.argmax(output_t)
        action_t[action_idx] = 1

        # decay exploration_rate
        if t > OBSERVE and exploration_rate > 0.05:
            exploration_rate -= 0.00002

        # Next state and reward
        x_new, r_new = sol.step(ACTIONS[action_idx])
        terminal = False # TODO

        D.append((x_t, r_new, action_t, x_new, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # restart game if over
        if terminal:
            sol.reset()

        # defer training until we got some data
        if t > OBSERVE:
            # Randomize the minibatch from replay memory
            minibatch = random.sample(D, 32)
            [x_js, r_news, action_ts, x_news, terminals] = zip(*minibatch)
            output_news = output.eval({x : x_news})
            ys = []
            for i in xrange(len(minibatch)):
                if terminals[i]:
                    ys.append(r_news[i])
                else:
                    ys.append(r_news[i] + 0.99 * np.max(output_news[i]))

            optimizer.run({x: x_js, y: ys, action: action_ts})


        # Save new values and increment the iteration counter
        x_t = x_new
        r_t = r_new
        t += 1

        # save weights
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/network', global_step = t)
        print t, exploration_rate, ACTIONS[action_idx], r_t, np.max(output_t)


def main():
    x, output = createNet()
    train(x, output)

if __name__ == "__main__":
    main()
