import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle


class TrainingStatus:
    def __init__(self, sess, num_epochs, num_batches, logwrite_interval = 25, eta_interval = 25, save_interval = 100, save_dir = "save", graph_def = None):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        if graph_def is not None:
            self.writer = tf.train.SummaryWriter(save_dir, graph_def)
        else:
            self.writer = tf.train.SummaryWriter(save_dir)

        self.save_dir = os.path.join(save_dir, 'model.ckpt')
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep = 0)

        lastCheckpoint = tf.train.latest_checkpoint(save_dir) 
        if lastCheckpoint is None:
            self.startEpoch = 0
        else:
            print "Last checkpoint :", lastCheckpoint
            self.startEpoch = int(lastCheckpoint.split("-")[-1])
            self.saver.restore(sess, lastCheckpoint)

        print "startEpoch = ", self.startEpoch

        self.logwrite_interval = logwrite_interval
        self.eta_interval = eta_interval
        self.totalTask = num_epochs * num_batches
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.save_interval = save_interval

        self.etaCount = 0
        self.etaStart = time.time()
        self.duration = 0

    def tic(self):
        self.start = time.time()

    def tocBatch(self, summary, e, b, loss):
        self.end = time.time()
        taskNum = (e * self.num_batches + b)

        if self.etaCount % self.logwrite_interval == 0:
            self.writer.add_summary(summary, taskNum)

        self.etaCount += 1

        if self.etaCount % self.eta_interval == 0:
            self.duration = time.time() - self.etaStart
            self.etaStart = time.time()

        etaTime = float(self.totalTask - (taskNum + 1)) / self.eta_interval * self.duration
        m, s = divmod(etaTime, 60)
        h, m = divmod(m, 60)
        etaString = "%d:%02d:%02d" % (h, m, s)

        return "%d/%d (epoch %d), loss = %.3f, time/batch = %.3f, ETA: %s (%s)" % (taskNum, self.totalTask, e, loss, self.end - self.start, time.strftime("%H:%M:%S", time.localtime(time.time() + etaTime)), etaString)

    def tocEpoch(self, sess, e):
        if (e + 1) % self.save_interval == 0 or e == self.num_epochs - 1:
            self.saver.save(sess, self.save_dir, global_step = e + 1)
            print "model saved to {}".format(self.save_dir)


