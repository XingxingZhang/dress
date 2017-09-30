#!/usr/bin/python2.7

import sys, os, random, argparse

def shuffle(infile, outfile):
    lines = open(infile).readlines()
    random.shuffle(lines)
    open(outfile, 'w').write(''.join(lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='shuffle lines of a file')
    parser.add_argument('input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-r', '--rand', type=int, default=7)
    args = parser.parse_args()

    random.seed(args.rand)
    if args.output:
        shuffle(args.input, args.output)
    else:
        output = '__tmp__'
        shuffle(args.input, output)
        os.system('cp %s %s' % (output, args.input))
        os.system('rm %s' % output)

