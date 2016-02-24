import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import numpy as np
import random

class Model():
  def __init__(self, args, infer=False):
    self.dim = 6
    self.args = args
    if infer:
      args.batch_size = 1
      args.seq_length = 1

    if args.model == 'rnn':
      cell_fn = rnn_cell.BasicRNNCell
    elif args.model == 'gru':
      cell_fn = rnn_cell.GRUCell
    elif args.model == 'lstm':
      cell_fn = rnn_cell.BasicLSTMCell
    else:
      raise Exception("model type not supported: {}".format(args.model))

    cell = cell_fn(args.rnn_size)

    cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

    if (infer == False and args.keep_prob < 1): # training mode
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = args.keep_prob)

    self.cell = cell

    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dim])
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dim])
    self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

    self.num_mixture = args.num_mixture
    NOUT = self.num_mixture * 3 * self.dim # prob + mu + sig

    with tf.variable_scope('rnnlm'):
      output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    inputs = tf.split(1, args.seq_length, self.input_data)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
    output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = states

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_data,[-1, self.dim])
    #[x1_data, x2_data, eos_data] = tf.split(1, 3, flat_target_data)
    x_data = flat_target_data

    def tf_normal(x, mu, sig):
        return tf.exp(-tf.square(x - mu) / (2 * tf.square(sig))) / (sig * tf.sqrt(2 * np.pi))

    
    def tf_multi_normal(x, mu, sig, ang):
        # use n (n+1) / 2 to parametrize covariance matrix
        # 1. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.494&rep=rep1&type=pdf
        # 2. https://en.wikipedia.org/wiki/Triangular_matrix
        # 3. https://makarandtapaswi.wordpress.com/2011/07/08/cholesky-decomposition-for-matrix-inversion/

        # A = LL'  by 1
        # det(L) = prod of diagonals  by 2
        # det(A) = det(L)^2  by 3
        # A-1 = (L-1)'(L-1)  by 3

        # We're parametrizing using L^-1
        # Sigma^-1 = (L^-1)'(L^-1)
        # |Sigma| = 1 / det(L^-1)^2 = 1 / (diagonal product of L^-1)^2
        return tf.exp(-tf.square(x - mu) / (2 * tf.square(sig))) / (sig * tf.sqrt(2 * np.pi))
        

    def get_lossfunc(z_pi, z_mu, z_sig, x_data):
      result0 = tf_normal(x_data, z_mu, z_sig) 
      result1 = tf.reduce_sum(result0 * z_pi, 1, keep_dims=True)
      result2 = -tf.log(tf.maximum(result1, 1e-20)) 
      return tf.reduce_sum(result2)

    # below is where we need to do MDN splitting of distribution params
    def get_mixture_coef(output):
        # output is 120
        # mu is 40
        z_pi, z_mu, z_sig = tf.split(1, 3, output)

        # softmax all the pi's:
        max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
        z_pi = tf.sub(z_pi, max_pi)
        z_pi = tf.exp(z_pi)
        normalize_pi = tf.inv(tf.reduce_sum(z_pi, 1, keep_dims=True))
        z_pi = tf.mul(normalize_pi, z_pi)

        z_sig = tf.exp(z_sig)
        return [z_pi, z_mu, z_sig]


    output_each_dim = tf.split(1, self.dim, output)
    #for i in range(self.dim):
    #x_da = tf.split(1, self.dim, x_data)
    self.pi = []
    self.mu = []
    self.sig = []
    self.cost = 0
    for i in range(self.dim):
        [o_pi, o_mu, o_sig] = get_mixture_coef(output_each_dim[i])

        self.pi.append(o_pi)
        self.mu.append(o_mu)
        self.sig.append(o_sig)

        lossfunc = get_lossfunc(o_pi, o_mu, o_sig, x_data[:,i:i+1])
        self.cost += lossfunc / (args.batch_size * args.seq_length * self.dim)

    self.pi = tf.concat(1, self.pi)
    self.mu = tf.concat(1, self.mu)
    self.sig = tf.concat(1, self.sig)

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))


  def sample(self, sess, num=1200):

    def get_pi_idx(x, pdf):
      N = pdf.size
      accumulate = 0
      for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
          return i
      print 'error with sampling ensemble'
      return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
      mean = [mu1, mu2]
      cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
      x = np.random.multivariate_normal(mean, cov, 1)
      return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 3), dtype=np.float32)
    prev_x[0, 0, 2] = 1 # initially, we want to see beginning of new stroke
    prev_state = sess.run(self.cell.zero_state(1, tf.float32))

    strokes = np.zeros((num, 3), dtype=np.float32)
    mixture_params = []

    for i in xrange(num):

      feed = {self.input_data: prev_x, self.initial_state:prev_state}

      [o_pi, o_mu, o_sig, next_state] = sess.run([self.pi, self.mu, self.sig, self.final_state],feed)
      idx = get_pi_idx(random.random(), o_pi[0])

      eos = 1 if random.random() < o_eos[0][0] else 0

      next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

      strokes[i,:] = [next_x1, next_x2, eos]

      params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_eos[0]]
      mixture_params.append(params)

      prev_x = np.zeros((1, 1, 3), dtype=np.float32)
      prev_x[0][0] = np.array([next_x1, next_x2, eos], dtype=np.float32)
      prev_state = next_state

    strokes[:,0:2] *= self.args.data_scale
    return strokes, mixture_params
