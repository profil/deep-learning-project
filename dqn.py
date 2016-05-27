#!/usr/bin/env python2

import tensorflow as tf
import numpy as np
import random
from solitaire import Solitaire
from collections import deque

ACTIONS = ['select', 'up', 'down', 'left', 'right']
REPLAY_MEMORY = 300000
OBSERVE = 20000

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def createConv():
    Wconv1 = weight([7, 7, 3, 32])
    bconv1 = bias([32])

    Wconv2 = weight([6, 6, 32, 64])
    bconv2 = bias([64])

    Wconv3 = weight([4, 4, 64, 64])
    bconv3 = bias([64])

    x = tf.placeholder(tf.float32, [None, 100, 100, 3])

    hconv1 = tf.nn.relu(tf.nn.conv2d(x, Wconv1, [1, 3, 3, 1], "VALID") + bconv1)

    hconv2 = tf.nn.relu(tf.nn.conv2d(hconv1, Wconv2, [1, 2, 2, 1], "VALID") + bconv2)

    hconv3 = tf.nn.relu(tf.nn.conv2d(hconv2, Wconv3, [1, 2, 2, 1], "VALID") + bconv3)

    tf.image_summary('conv1', tf.transpose(hconv1, [3, 1, 2, 0]), 32)
    tf.image_summary('weights', tf.transpose(Wconv1, [3, 0, 1, 2]), 32)

    saver = tf.train.Saver([Wconv1, bconv1, Wconv2, bconv2, Wconv3, bconv3])
    return x, hconv3, saver

def createFCsoftmax(hconv3):
    Wfc1 = weight([2304, 512])
    bfc1 = bias([512])

    Wfc2 = weight([512, 52])
    bfc2 = bias([52])

    hfc1 = tf.nn.relu(tf.matmul(tf.reshape(hconv3, [-1, 2304]), Wfc1) + bfc1)

    output = tf.matmul(hfc1, Wfc2) + bfc2

    saver = tf.train.Saver([Wfc1, bfc1, Wfc2, bfc2])

    return output, saver


def createFC(hconv3):
    Wfc1 = weight([2304, 512])
    bfc1 = bias([512])

    Wfc2 = weight([512, 5])
    bfc2 = bias([5])

    hfc1 = tf.nn.relu(tf.matmul(tf.reshape(hconv3, [-1, 2304]), Wfc1) + bfc1)

    output = tf.matmul(hfc1, Wfc2) + bfc2

    return output

def train_cards(x, output, conv_saver, fc_saver):
    sol = Solitaire()
    sess = tf.InteractiveSession()
    y = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, y)
    optimizer = tf.train.RMSPropOptimizer(1e-6, momentum=0.95).minimize(cross_entropy)

    sess.run(tf.initialize_all_variables())

    checkpoint = tf.train.get_checkpoint_state("conv_weights")
    if checkpoint and checkpoint.model_checkpoint_path:
        conv_saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old conv weights"

    checkpoint = tf.train.get_checkpoint_state("fc_weights")
    if checkpoint and checkpoint.model_checkpoint_path:
        fc_saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old fc weights"

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('summaries_cards/', sess.graph)

    t = 0
    a = 5
    while True:
        b = 0
        batch = []
        while b < 32:
            if b % a == 0:
                sol.reset()
                x_t, r_t, terminal = sol.step('down')
            card = sol.deck.rows[b % a].cards[-1]
            value = card.suit * 13 + card.value - 1
            batch.append((x_t, value))
            b += 1
            x_t, r_t, terminal = sol.step('right')

        t += 1
        [xs, ys] = zip(*(random.sample(batch, 32)))
        optimizer.run({x: xs, y: ys})

        if t % 3000 == 0:
            a += 1
            if a > 7:
                a = 5
        if t % 10000 == 0:
            conv_saver.save(sess, 'conv_weights/saved', global_step = t)
            fc_saver.save(sess, 'fc_weights/saved', global_step = t)
	    summary_writer.add_summary(sess.run(summary_op, {x: [xs[0]], y: [ys[0]]}), global_step = t)
        if t % 10 == 0:
            output_t = output.eval({x: [xs[0]], y: [ys[0]]})[0]
            print('{:>8} {:>1} {:>2} {:>2} {:>8.8}'.format(*[t, a, np.argmax(output_t) , ys[0], sess.run(cross_entropy, {x: [xs[0]], y: [ys[0]]})[0]]))

def train(x, output):
    sol = Solitaire()
    sess = tf.InteractiveSession()

    episode = 1
    loss_avg = 0
    loss_total = 0
    output_avg = 0
    output_total = 0

    action = tf.placeholder(tf.float32, [None, 5])
    y = tf.placeholder(tf.float32, [None])
    output_action = tf.reduce_sum(tf.mul(output, action), 1)
    loss = tf.reduce_mean(tf.square(tf.clip_by_value(y - output_action, -1, 1)))
    optimizer = tf.train.RMSPropOptimizer(1e-6).minimize(loss)

    tf.scalar_summary('average/loss', loss_avg)
    tf.scalar_summary('average/output', output_avg)

    D = deque()

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('summaries/', sess.graph)

    while True:
        sol.reset()
        x_t, r_t, terminal = sol.step()
        exploration_rate = 1.0
        summary_writer.add_summary(sess.run(summary_op, {x: [x_t]}), global_step = episode)

        t = 1
        while t < 30000:
            action_t = np.zeros([5])
            output_t = None

            # Sometimes (according to the exploration_rate)
            # pick an entirely random action instead of the
            # best prediction.
            if random.random() <= exploration_rate or t < OBSERVE:
                action_idx = ACTIONS.index(sol.bot())
            else:
                output_t = output.eval({x: [x_t]})[0]
                action_idx = np.argmax(output_t)
            action_t[action_idx] = 1

            # decay exploration_rate
            if t > OBSERVE and exploration_rate > 0.05:
                exploration_rate -= 0.00005

            # Next state and reward
            x_new, r_new, terminal = sol.step(ACTIONS[action_idx])

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

                _, loss_t, output_ts = sess.run([optimizer, loss, output], {x: x_js, y: ys, action: action_ts})
                loss_total += loss_t
                output_total += output_ts.mean()
                loss_avg = loss_total/ (t * episode - OBSERVE + 1)
                output_avg = output_total / (t * episode - OBSERVE + 1)

            # Save new values and increment the iteration counter
            x_t = x_new
            r_t = r_new
            t += 1

            # save weights
            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/network', global_step = t)
            print('{:>5} {:>5} {:<8} {:>6} {:>2} {:<8f} {:<8f}'.format(*[episode, t, exploration_rate, ACTIONS[action_idx], r_t, loss_avg, output_avg]))

        episode += 1


def main():
    x, hconv3, conv_saver = createConv()
    output = createFC(hconv3)
    train(x, output)
    #output, fc_saver = createFCsoftmax(hconv3)
    #train_cards(x, output, conv_saver, fc_saver)

if __name__ == "__main__":
    main()
