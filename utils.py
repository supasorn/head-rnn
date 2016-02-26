import os
import glob
import cPickle
import numpy as np 
import random
import math

class DataLoader():
  def __init__(self, batch_size=50, seq_length=300, scale_factor = 1, limit = 500, reprocess = 0):
    self.data_dir = "./data"
    self.pose_dir = "/home/supasorn/face-singleview/data/Obama2/"

    self.batch_size = batch_size
    self.seq_length = seq_length
    self.scale_factor = scale_factor # divide data by this factor
    self.limit = limit # removes large noisy gaps in the data

    data_file = os.path.join(self.data_dir, "training.cpkl")

    if not (os.path.exists(data_file)) or reprocess:
        print "creating training data cpkl file from raw source"
        self.preprocess(self.pose_dir, data_file)

    self.load_preprocessed(data_file)
    self.reset_batch_pointer()

  def rodriguesToEuler(self, a, b, c):
    mat = self.rodriguesToMatrix(a, b, c)
    x = math.asin(-mat[1, 2])
    if mat[1, 2] == -1:
      z = 0
      y = math.atan2(mat[0, 1], mat[0, 0])
    elif mat[1, 2] == 1:
      z = 0
      y = math.atan2(-mat[0, 1], mat[0, 0])
    else:
      cos2 = math.cos(x)
      z = math.atan2(mat[1, 0]/cos2, mat[1, 1]/cos2)
      y = math.atan2(mat[0, 2]/cos2, mat[2, 2]/cos2)
    return x, y, z

  def rodriguesToMatrix(self, x, y, z):
    axis = np.asarray([x, y, z])
    theta = np.linalg.norm(axis) 
    axis /= theta

    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

  def preprocess(self, data_dir, data_file):
    # create data file from raw xml files from iam handwriting source.

    files = [x.split("\t")[0] for x in open(data_dir + "processed.txt", "r").readlines()]
    filelist = []
    for i in range(len(files)):
        dnums = sorted([os.path.basename(x) for x in glob.glob(data_dir  + files[i] + "}}*")])
        for dnum in dnums:
            filelist.append(dnum)


    # function to read each individual xml file
    def getStrokes(filename):
        print "in", filename
        num = 1
        result = []
        while os.path.exists(filename + "/%04d.txt" % num):
            f = open(filename + "/%04d.txt" % num, "r")
            pose = [float(x) for x in f.read().strip().split(" ")[:6]]
            # convert rodrigues to euler
            pose[:3] = self.rodriguesToEuler(pose[0], pose[1], pose[2])

            result.append(pose)
            num += 1
        return result

    # converts a list of arrays into a 2d numpy int16 array
    def convert_stroke_to_array(stroke):

        n_point = 0
        stroke_data = np.zeros((len(stroke), len(stroke[0])), dtype=np.float32)

        counter = 0
        prev = [0] * len(stroke[0])
        for j in range(0, len(stroke)):
            for k in range(len(stroke[j])):
                stroke_data[counter, k] = float(stroke[j][k]) - prev[k]
                prev[k] = float(stroke[j][k])
            counter += 1

        #counter = 0
        #prev = stroke[0]
        #for j in range(0, len(stroke)):
            #for k in range(len(stroke[j])):
                #stroke_data[counter, k] = float(stroke[j][k]) - prev[k]
                #prev[k] = float(stroke[j][k])
            #counter += 1

        print stroke_data
        return stroke_data

    # build stroke database of every xml file inside iam database
    strokes = []
    for i in range(len(filelist)):
        if os.path.exists(data_dir + filelist[i] + "/poses2/"):
            print 'processing '+filelist[i]
            strokes.append(convert_stroke_to_array(getStrokes(data_dir + filelist[i] + "/poses2/")))

    f = open(data_file,"wb")
    cPickle.dump(strokes, f, protocol=2)
    f.close()


  def load_preprocessed(self, data_file):
    f = open(data_file,"rb")
    self.raw_data = cPickle.load(f)
    f.close()

    # goes thru the list, and only keeps the text entries that have more than seq_length points
    self.data = []
    counter = 0

    for i, data in enumerate(self.raw_data):
      f = open("data/%04d.txt" % i, "w")
      if len(data) >= (self.seq_length+2):
        data *= self.scale_factor
        self.data.append(data[:, :3])
        counter += int(len(data) / ((self.seq_length+2))) 
        for j in range(len(data)):
            f.write(" ".join(["%010.5F" % x for x in data[j]]) + "\n")
      f.close()

    self.num_batches = int(counter / self.batch_size)

  def next_batch(self):
    # returns a randomised, seq_length sized portion of the training data
    x_batch = []
    y_batch = []
    for i in xrange(self.batch_size):
      data = self.data[self.pointer]
      n_batch = int(len(data) / ((self.seq_length+2))) 

      idx = random.randint(1, len(data) - self.seq_length - 1)
      x_batch.append(np.copy(data[idx:idx+self.seq_length]))
      y_batch.append(np.copy(data[idx+1:idx+self.seq_length+1]))

      if random.random() < (1.0/float(n_batch)): 
        self.tick_batch_pointer()
    return x_batch, y_batch

  def tick_batch_pointer(self):
    self.pointer += 1
    if (self.pointer >= len(self.data)):
      self.pointer = 0

  def reset_batch_pointer(self):
    self.pointer = 0

