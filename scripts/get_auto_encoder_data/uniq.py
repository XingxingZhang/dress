#!/usr/bin/python

import sys

def uniq(infile,outfile):
    s = set()
    fout = open(outfile, 'w')
    for line in open(infile):
        if not line in s:
            fout.write(line)
            s.add(line)
    fout.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'delete the duplicated lines in <infile> and output <outfile>'
        print 'uniq.py <infile> <outfile>'
        sys.exit(1)
    uniq(sys.argv[1], sys.argv[2])
