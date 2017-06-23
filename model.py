# This is the file that has the class MemN2N

import tensorflow as tf # ML
import numpy as np # matrix handling
import math # math functions
from six.moves import xrange # saves memory over range
import random # random
import os # porting with os
import matplotlib.pyplot as plt

class MemN2N(object):
	"""docstring for MemN2N"""
	def __init__(self, config, session):
		# Dimensions
		self.lin_dim = config.lin_dim
		self.e_dim = config.e_dim # dimension of embedding
		self.n_words = config.n_words # number of words / size of vocab
		self.mem_size = config.mem_size # size of memory
		self.batch_size = config.batch_size # batch size for training

		self.total_correct = 0 # This is the counter for total correct examples

		# hyper parameters for proper tuning
		self.init_stddev = config.init_stddev # initial standard deviation for matrix generation
		self.n_hops = config.n_hops # number of hops to perform
		self.n_epochs = config.n_epochs # total number of epochs
		self.max_grad_norm = config.max_grad_norm # used in clipping of gradients
		self.current_lr = config.init_lr # initial learning rate
		self.n_hidden = config.n_hidden # the number of hidden units in the last layer
		self.keep_prob = config.keep_prob # keeping probability for the dropout layer

		# model running mode
		self.test_true = config.test_true # if true model is in testing mode
		self.d_steps = config.d_steps # number of steps after which results are displyed
		self.save_epoch = config.save_epoch # To save 

		# checkpoint directory check
		self.checkpoint_dir = config.checkpoint_dir # the checkpoint directory to load the model
		if not os.path.isdir(self.checkpoint_dir):
			raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

		# tensorflow ops
		self.sess = session # the current tensorflow session
		self.input_t = tf.placeholder(tf.float32, shape = [None, self.e_dim], name = 'input_t') # input tensor
		self.time = tf.placeholder(tf.int32, shape = [None, self.mem_size], name = 'time') # time tensor for using temporal encoding
		self.memory = tf.placeholder(tf.int32, shape = [None, self.mem_size], name = 'memory') # memory tensor
		self.labels = tf.placeholder(tf.float32, shape = [None, self.n_words], name = 'labels') # target labels tensor

		# basic hidden variables
		self.init_hidden = config.init_hidden # the initial hidden state
		# self.hidden = self.input_t # hidden state
		self.hidden = []
		self.hidden.append(self.input_t) # add the input to the hidden state
		self.store_list = [] # a list of lists to store misc
		self.store_list.append([]) # we store the values of b_out in the first list
		self.train_acc_log = [] # training accuracy log
		self.valid_acc_log = [] # testing accuracy log

		self.epoch_exec = 0 # Since we are using annealing, the total number of epochs run is not equal to n_epochs

		# Misc. functions
		self.loss = None # loss function, globally called
		self.lr = None # learnign rate, globally called
		self.step = None # step variable, globally called
		self.optim = None # optimizer, globally called
		self.saver = None # saver function from tensorflow, globally called

	def build(self):
		# This is the function that is to be used once and will be used to make the memory
		self.global_step = tf.Variable(0, name = 'global_step')

		# making embeding matrix and temporal emcoding matrix pair A
		self.A = tf.Variable(tf.truncated_normal(shape = [self.n_words, self.e_dim], stddev = self.init_stddev), name = 'A')
		self.Temp_A = tf.Variable(tf.truncated_normal(shape = [self.mem_size, self.e_dim], stddev = self.init_stddev), name = 'Temp_A')
		# making m_i = sum(A_ij, x_ij) + Temp_A_i
		# lookup from memory and time, both these things improve with training
		a_con = tf.nn.embedding_lookup(self.A, self.memory, name = 'a_con') # (batch_size, mem_size, e_dim)
		a_temp = tf.nn.embedding_lookup(self.Temp_A, self.time, name = 'a_temp') # (batch_size, mem_size, e_dim)
		a_pre_hid = tf.add(a_con, a_temp, name = 'a_pre_hid') # (batch_size, mem_size, e_dim)
		
		# making embeding matrix and temporal emcoding matrix pair C
		self.C = tf.Variable(tf.truncated_normal(shape = [self.n_words, self.e_dim], stddev = self.init_stddev), name = 'C')
		self.Temp_C = tf.Variable(tf.truncated_normal(shape = [self.mem_size, self.e_dim], stddev = self.init_stddev), name = 'Temp_C')
		# making c_i = sum(C_ij, x_ij) + Temp_C
		# lookup from memory and time, both these things improve with training
		c_con = tf.nn.embedding_lookup(self.C, self.memory, name = 'c_con') # (batch_size, mem_size, e_dim)
		c_temp = tf.nn.embedding_lookup(self.Temp_C, self.time, name = 'c_temp') # (batch_size, mem_size, e_dim)
		c_pre_hid = tf.add(c_con, c_temp, name = 'c_pre_hid') # (batch_size, mem_size, e_dim)

		# making embedding matrix B
		self.B = tf.Variable(tf.truncated_normal(shape = [self.e_dim, self.e_dim], stddev = self.init_stddev), name = 'B')

		for _ in xrange(self.n_hops):
			# first we reshape the hidden state
			self.hidden_3_dim = tf.reshape(self.hidden[-1], [-1, 1, self.e_dim], name = 'hidden_3_dim') # (batch_size, 1, e_dim)

			# getting the output of A
			a_out = tf.matmul(self.hidden_3_dim, a_pre_hid, adjoint_b = True, name = 'a_out') # (batch_size, 1, mem_size)
			a_out_2_dim = tf.reshape(a_out, [-1, self.mem_size], name = 'a_out_2_dim') # (batch_size, mem_size)
			p = tf.nn.softmax(a_out_2_dim, name = 'p') # (batch_size, mem_size)

			# Once we ge the probability distribution we use it to find output from c
			p_3_dim = tf.reshape(p, [-1, 1, self.mem_size], name = 'p_3_dim') # (batch_size, 1, mem_size)
			c_out = tf.matmul(p_3_dim, c_pre_hid, name = 'c_out') # (batch_size, 1, e_dim)
			c_out_2_dim = tf.reshape(c_out, [-1, self.e_dim]) # (batch_size, e_dim)

			# using the input to directly get the output
			b_out = tf.matmul(self.hidden[-1], self.B, name = 'b_out') # (batch_size, e_dim)
			# final output from the memory module
			mem_out = tf.add(b_out, c_out_2_dim, name = 'mem_out') # (batch_size, e_dim)

			# Storing the value of output of b to the store list
			self.store_list[0].append(b_out)

			# adding to the hidden list the output of the memory
			self.hidden.append(mem_out)

		# Now making rest of the model

		# UPDATE: 22/07/16/0024
		# adding a line of code that adds the input question to the output from the memory, as sugegsted by the paper
		# self.hidden += self.input_t

		# Adding a neural network here
		self.W1 = tf.Variable(tf.truncated_normal(shape = [self.e_dim, self.n_hidden], stddev = self.init_stddev), name = 'W1') # the final weight
		self.pre_f = tf.matmul(self.hidden[-1], self.W1, name = 'pre_f') # (batch_size, hidden)
		# self.pre_f = tf.nn.tanh(self.pre_f)

		# Adding a dropout layer
		# self.dropout_op = tf.nn.dropout(self.pre_f, keep_prob = self.keep_prob, name = 'dropout_op')

		# Adding another layer
		self.W2 = tf.Variable(tf.truncated_normal(shape = [self.n_hidden, self.n_words], stddev = self.init_stddev), name = 'W2')
		self.final_op = tf.matmul(self.pre_f, self.W2, name = 'final_op') # (batch_size, n_words)
		# self.final_op = tf.nn.softmax(self.final_op)

		# determining the loss function
		# self.loss = tf.reduce_sum(tf.pow(self.labels - self.final_op, 2)/(2 * self.batch_size))
		self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.final_op, labels = self.labels, name = 'loss')

		# Declaring hyper parameters
		self.lr = tf.Variable(self.current_lr)
		'''
		# optimizer
		self.opt = tf.train.GradientDescentOptimizer(self.lr)
		'''

		# Using a conventional Adam, using the standard values
		self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		'''
		# parameters
		parameters = [self.A, self.C, self.B, self.Temp_A, self.Temp_C, self.W1, self.W2] 
		grad_and_vars = self.opt.compute_gradients(self.loss,parameters)
		clipped_grad_and_vars = [(tf.clip_by_norm(g_and_v[0], self.max_grad_norm), g_and_v[1]) for g_and_v in grad_and_vars]

		inc = self.global_step.assign_add(1)
		with tf.control_dependencies([inc]):
			# inc is an operation that must be completed and then we can proceed to the next step
			# apply the gradients to the opt and pass it as the optimizer
			self.optim = self.opt.apply_gradients(clipped_grad_and_vars)
		'''
		# initilaizing the variables
		tf.global_variables_initializer().run()
		self.saver = tf.train.Saver()

	def train(self, data):
		# Determining the number of batches
		n_batches = int(math.ceil(len(data) / self.batch_size))
		cost = 0

		time = np.ndarray([self.batch_size, self.mem_size], dtype = np.float32)

		for t in xrange(self.mem_size):
			time[:,t].fill(t)

		correct = 0 # number of correct options

		for idx in xrange(n_batches):
			acc_batch = 0

			memory = np.array([item[0] for item in data[idx*self.batch_size: (idx+1)*self.batch_size]])
			input_x = np.array([item[1] for item in data[idx*self.batch_size: (idx+1)*self.batch_size]])
			label = np.array([item[2] for item in data[idx*self.batch_size: (idx+1)*self.batch_size]])

			_, self.step, op = self.sess.run([self.optim, self.global_step, self.final_op],
				feed_dict={ self.input_t: input_x, self.time: time, self.labels: label, self.memory: memory})
			
			for corr_idx in xrange(len(label)):
				# print(corr_idx)
				corr_arg_op = np.argmax(op[corr_idx])
				corr_arg_lb = np.argmax(label[corr_idx])
				if corr_arg_op == corr_arg_lb:
					acc_batch += 1

			acc_batch /= self.batch_size
			correct += acc_batch

		return correct /(n_batches)

	def test(self, data):
		# Determining the number of batches
		n_batches = int(math.ceil(len(data) / self.batch_size))
		cost = 0

		time = np.ndarray([self.batch_size, self.mem_size], dtype = np.float32)

		for t in xrange(self.mem_size):
			time[:,t].fill(t)

		correct = 0 # correct counter for accuracy

		for idx in xrange(n_batches):
			acc_test = 0

			memory = np.array([item[0] for item in data[idx*self.batch_size: (idx+1)*self.batch_size]])
			input_x = np.array([item[1] for item in data[idx*self.batch_size: (idx+1)*self.batch_size]])
			label = np.array([item[2] for item in data[idx*self.batch_size: (idx+1)*self.batch_size]])

			op = self.sess.run([self.final_op],
				feed_dict={ self.input_t: input_x, self.time: time, self.labels: label, self.memory: memory})
			for corr_idx in xrange(len(label)):
				corr_arg_op = np.argmax(op[0][corr_idx])
				corr_arg_lb = np.argmax(label[corr_idx])
				if corr_arg_op == corr_arg_lb:
					acc_test += 1

			acc_test /= self.batch_size
			correct += acc_test

		return correct/(n_batches)

	def run(self, train_data, test_data):
		if not self.test_true:
			for epoch in xrange(self.n_epochs):
				self.epoch_exec += 1
				train_acc = self.train(train_data)
				valid_acc = self.test(test_data)

				# logging
				self.train_acc_log.append(train_acc)
				self.valid_acc_log.append(valid_acc)

				# Defining the current state
				state = {
					'epoch': epoch,
					'valid_acc': valid_acc,
					'train_acc': train_acc,
				}

				if epoch % self.d_steps == 0:
					print(state)

				'''
				# learning rate annealing (according to the way given in paper)
				if len(self.loss_log) > 1 and self.loss_log[-1][1] > self.loss_log[-2][1] * 0.9999:
					self.current_lr = self.current_lr / 1.5
					self.lr.assign(self.current_lr).eval()
				if self.current_lr < 1e-5:
					# when to break from running
					break
				'''

				'''
				# learning rate exactly the way given in paper
				if len(self.loss_log) > 1 and epoch % 25 == 0:
					self.current_lr = self.current_lr/1.1
					self.lr.assign(self.current_lr).eval()

				if self.current_lr < 1e-6:
					break
				'''
				if epoch % self.save_epoch == 0:
					self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "MemN2N.model"), global_step = self.step.astype(int))

		else:
			self.load_model()

			# validation_loss = np.sum(self.test(train_data, label='Validation')[0])
			# test_loss = np.sum(self.test(test_data, label='Test')[0])
			test_acc = self.test(test_data)

			state = {
				'accuracy': test_acc
			}
			print(state)

	def show_plot(self):
		y = [item[1] for item in self.train_acc_log]
		x = [i for i in range(self.epoch_exec)]
		plt.plot(x, y)
		plt.show()

	def load_model(self):
		print("[*] Reading checkpoints...")
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path: 
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			raise Exception("[!] Test mode but no checkpoint found")
