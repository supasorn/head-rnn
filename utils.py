import os
import glob
import cPickle
import numpy as np 
import random
import math

class DataLoader():
  def __init__(self, dim = 3, batch_size=50, seq_length=300, reprocess = 0):
    self.data_dir = "./data"
    #self.pose_dir = "/home/supasorn/face-singleview/data/Obama2/"
    self.pose_dir = "/projects/grail/supasorn2nb/face-singleview/data/Obama2/"

    self.dim = dim
    self.batch_size = batch_size
    self.seq_length = seq_length

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
    f = open("/projects/grail/supasorn2nb/face-singleview/data/Obama2/pose_training.txt", "r")
    sumn = 0

    strokes = []
    
    while True:
        dnum = f.readline()
        if not dnum: break
        n = int(f.readline())
        sumn += n
        stroke = np.zeros((n, self.dim), dtype=np.float32)
        print dnum, n
        for i in range(n):
            st = f.readline().split(" ")
            stroke[i, :3] = np.array(self.rodriguesToEuler(float(st[1]), float(st[2]), float(st[3])))
            for j in range(3, self.dim):
                stroke[i, j] = st[j + 1]
            
                
        strokes.append(np.multiply(stroke, np.array([100, 100, 100, 1, 1, 0.001])))

    f.close()
    print sumn

    f = open(data_file, "wb")
    cPickle.dump(strokes, f, protocol=2)
    f.close()

    np.savetxt("data/data.txt", np.concatenate(strokes, 0), "%.3f")


  def load_preprocessed(self, data_file):
    f = open(data_file,"rb")
    self.raw_data = cPickle.load(f)
    f.close()

    # goes thru the list, and only keeps the text entries that have more than seq_length points
    self.data = []
    counter = 0

    for data in self.raw_data:
      if len(data) >= (self.seq_length+2):
        diff = data - np.concatenate([np.zeros((len(data), 3)), np.concatenate([data[:1,3:], data[:-1,3:]])], axis=1)
        self.data.append(diff)
        #self.data.append(data)
        #print data
        #print diff
        #exit(0)

        counter += int(len(data) / ((self.seq_length+2))) 
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

