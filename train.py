import sys
execfile(sys.path[0] + "/tensorutils.py")

from utils import DataLoader
from model import Model

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_size', type=int, default=256,
                     help='size of RNN hidden state')
<<<<<<< HEAD
  parser.add_argument('--num_layers', type=int, default=6,
=======
  parser.add_argument('--num_layers', type=int, default=4,
>>>>>>> 37fd518ed7c7722cecb35502effb9b59820e49a4
                     help='number of layers in the RNN')
  parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
  parser.add_argument('--batch_size', type=int, default=50,
                     help='minibatch size')
  parser.add_argument('--seq_length', type=int, default=300,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=50,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=10,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
<<<<<<< HEAD
  parser.add_argument('--decay_rate', type=float, default=0.98,
=======
  parser.add_argument('--decay_rate', type=float, default=0.97,
>>>>>>> 37fd518ed7c7722cecb35502effb9b59820e49a4
                     help='decay rate for rmsprop')
  parser.add_argument('--num_mixture', type=int, default=20,
                     help='number of gaussian mixtures')
  parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
  parser.add_argument('--reprocess', type=int, default=0,
                     help='reprocess input')
  args = parser.parse_args()
  train(args)

def train(args):
    dim = 6
    data_loader = DataLoader(dim, args.batch_size, args.seq_length, reprocess=args.reprocess)
    x, y = data_loader.next_batch()

    with open(os.path.join('save', 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    model = Model(dim, args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        ts = TrainingStatus(sess, args.num_epochs, data_loader.num_batches, save_interval = args.save_every, graph_def = sess.graph_def)
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

if __name__ == '__main__':
  main()


