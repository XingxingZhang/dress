
import os, sys

def load(infile):
	s = set()
	for line in open(infile):
		s.add(line)
	return s


def check(train_file, valid_file):
	valid = load(valid_file)
	for line in open(train_file):
		if line in valid:
			print line



check('train.txt', 'valid.txt')
check('train.txt', 'test.txt')
check('valid.txt', 'test.txt')




