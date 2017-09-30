
import os, sys, argparse

def filter(train_file, valid_file, out_file):
	valid_set = set(open(valid_file).readlines())
	fout = open(out_file, 'w')
	cnt_del = 0
	cnt = 0
	for line in open(train_file):
		if not line in valid_set:
			fout.write(line)
		else:
			cnt_del += 1
		cnt += 1
	
	fout.close()

	print 'del %d/%d = %f' % (cnt_del, cnt, float(cnt_del)/cnt)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help = 'train file')
	parser.add_argument('--valid', help = 'valid file')
	parser.add_argument('--out', help = 'out file')

	args = parser.parse_args()

	filter(args.train, args.valid, args.out)




