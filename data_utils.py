from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
	'''Return the tokens of a sentence including punctuation.
	>>> tokenize('Bob dropped the apple. Where is the apple?')
	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	'''
	return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
	'''Parse stories provided in the bAbi tasks format
	If only_supporting is true,
	only the sentences that support the answer are kept.
	'''
	data = []
	story = []
	for line in lines:
		line = line.decode('utf-8').strip()
		nid, line = line.split(' ', 1)
		nid = int(nid)
		if nid == 1:
			story = []
		if '\t' in line:
			q, a, supporting = line.split('\t')
			q = tokenize(q)
			substory = None
			if only_supporting:
				# Only select the related substory
				supporting = map(int, supporting.split())
				substory = [story[i - 1] for i in supporting]
			else:
				# Provide all the substories
				substory = [x for x in story if x]
			data.append((substory, q, a))
			story.append('')
		else:
			sent = tokenize(line)
			story.append(sent)
	return data


def get_stories(f, only_supporting=False, max_length=None):
	'''Given a file name, read the file, retrieve the stories,
	and then convert the sentences into a single story.
	If max_length is supplied,
	any stories longer than max_length tokens will be discarded.
	'''
	data = parse_stories(f.readlines(), only_supporting=only_supporting)
	flatten = lambda data: reduce(lambda x, y: x + y, data)
	data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
	return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
	xs = []
	xqs = []
	ys = []
	for story, query, answer in data:
		x = [word_idx[w] for w in story]
		xq = [word_idx[w] for w in query]
		# let's not forget that index 0 is reserved
		y = np.zeros(len(word_idx) + 1)
		y[word_idx[answer]] = 1
		xs.append(x)
		xqs.append(xq)
		ys.append(y)
	return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)

def get_data(challenge):
	try:
		path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
	except:
		print('Error downloading dataset, please download it manually:\n'
			'$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
			'$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
		raise
	
	tar = tarfile.open(path)

	train = get_stories(tar.extractfile(challenge.format('train')))
	test = get_stories(tar.extractfile(challenge.format('test')))

	vocab = set()
	for story, q, answer in train + test:
		vocab |= set(story + q + [answer])
	vocab = sorted(vocab)

	# Reserve 0 for masking via pad_sequences
	vocab_size = len(vocab) + 1
	word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
	story_maxlen = max(map(len, (x for x, _, _ in train + test)))
	query_maxlen = max(map(len, (x for _, x, _ in train + test)))

	train_s, train_q, train_a = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
	test_s, test_q, test_a = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

	data_train = []
	for i in range(len(train_s)):
		s = train_s[i]
		q = train_q[i]
		a = train_a[i]
		data_train.append((s, q, a))

	data_test = []
	for i in range(len(test_s)):
		s = test_s[i]
		q = test_q[i]
		a = test_a[i]
		data_test.append((s, q, a))
		
	return data_train, data_test