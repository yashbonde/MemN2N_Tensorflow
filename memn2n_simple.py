import os
import pprint
import tensorflow as tf
from model import MemN2N
from data_utils import get_data

pp = pprint.PrettyPrinter()
flags = tf.app.flags

flags.DEFINE_integer('mem_size', 86, 'size of memory(curr = max_len_s)')
flags.DEFINE_integer('e_dim', 4, 'embedding dimension(curr = max_len_q)')
flags.DEFINE_integer('lin_dim', 0, 'linear embedding dimension')
flags.DEFINE_integer('n_words', 23, 'size of vocab(22)')
flags.DEFINE_integer('batch_size', 50, 'size of batch during training')
flags.DEFINE_float('init_stddev', 0.1, 'initial standard deviation')
flags.DEFINE_integer('n_hops', 3, 'number of hops')
flags.DEFINE_integer('n_epochs', 200, 'number of epochs')
flags.DEFINE_float('max_grad_norm', 50, 'clip gradients to this norm')
flags.DEFINE_float('init_lr', 0.02, 'initial learning rate')
flags.DEFINE_boolean('test_true', False, 'True for testing')
flags.DEFINE_float('init_hidden', 0.1, 'initial hidden state value')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'checkpoint directory [checkpoints]')
flags.DEFINE_string('data_name', 'bAbI tasks', 'data set name')
flags.DEFINE_float('train_size', 0.9, 'ratio of training data to whole data')
flags.DEFINE_float('valid_size', 0.1, 'ratio of validation data to whole date')
flags.DEFINE_float('test_size', 0.1, 'ratio of test data to whole date')
flags.DEFINE_integer('d_steps', 20, 'number of steps after which results are displyed')
flags.DEFINE_boolean('show_plot', False, 'if true shows the plot of loss vs. epochs')
flags.DEFINE_integer('n_hidden', 50, 'number of hidden units')
flags.DEFINE_float('keep_prob', 0.90, 'probability for dropout layer')
flags.DEFINE_integer('save_epoch', 100, 'after these epochs model is saved')

FLAGS = flags.FLAGS

def main(_):
	challenge = 'tasks_1-20_v1-2/en/qa12_conjunction_{}.txt'
	data_train, test_data = get_data(challenge)
	train_data = data_train[:int(FLAGS.train_size * len(data_train))]
	valid_data = data_train[int(FLAGS.train_size * len(data_train)):]

	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)

	# pp.pprint(flags.FLAGS.__flags)

	with tf.Session() as sess:
		model = MemN2N(FLAGS, sess)
		model.build()

		if not FLAGS.test_true:
			model.run(train_data, valid_data)
			if FLAGS.show_plot:
				model.show_plot()
		
		else:
			model.run(valid_data, test_data)

if __name__ == '__main__':
    tf.app.run()