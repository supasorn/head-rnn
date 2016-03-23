import sys
execfile(sys.path[0] + "/tensorutils.py")

from utils import DataLoader
from model import Model
import json

def main():
  np.random.seed(42)
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_size', type=int, default=256,
                     help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=4,
                     help='number of layers in the RNN')
  parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
  parser.add_argument('--batch_size', type=int, default=50,
                     help='minibatch size')
  parser.add_argument('--seq_length', type=int, default=300,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=100,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=10,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=0.97,
                     help='decay rate for rmsprop')
  parser.add_argument('--num_mixture', type=int, default=20,
                     help='number of gaussian mixtures')
  parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
  parser.add_argument('--save_dir', type=str, default='save',
                     help='dropout keep probability')
  parser.add_argument('--sample_length', type=int, default=500,
                     help='number of strokes to sample')

  parser.add_argument('--sample', action='store_true')
  parser.add_argument('--reprocess', action='store_true')
  args = parser.parse_args()

  if args.sample:
    sample(args)
  else:
    train(args)

def train(args):
  dim = 6
  data_loader = DataLoader(dim, args.batch_size, args.seq_length, reprocess=args.reprocess)
  x, y = data_loader.next_batch()

  if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

  with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
    cPickle.dump(args, f)

  model = Model(dim, args)

  with tf.Session() as sess:
    tf.initialize_all_variables().run()

    ts = TrainingStatus(sess, args.num_epochs, data_loader.num_batches, save_interval = args.save_every, graph_def = sess.graph_def, save_dir = args.save_dir)
    for e in xrange(ts.startEpoch, args.num_epochs):
      sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
      data_loader.reset_batch_pointer()
      state = model.initial_state.eval()
      for b in xrange(data_loader.num_batches):
        ts.tic()
        x, y = data_loader.next_batch()
        feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
        summary, train_loss, state, _ = sess.run([model.summary, model.cost, model.final_state, model.train_op], feed)
        print ts.tocBatch(summary, e, b, train_loss)
        
      ts.tocEpoch(sess, e)

def sample(args):
  with open(os.path.join(args.save_dir, 'config.pkl')) as f:
    saved_args = cPickle.load(f)

  model = Model(6, saved_args, True)
  sess = tf.InteractiveSession()
  saver = tf.train.Saver()

  ckpt = tf.train.get_checkpoint_state(args.save_dir)
  print "loading model: ",ckpt.model_checkpoint_path

  saver.restore(sess, ckpt.model_checkpoint_path)
  model.sample(sess, args.sample_length)


if __name__ == '__main__':
  main()


