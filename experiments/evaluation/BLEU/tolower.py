#!/usr/bin/python2.7

import argparse

def getArgs():
    # tolower.py infile outfile
    parser = argparse.ArgumentParser("convert a text file into lower case")
    parser.add_argument('input', type=argparse.FileType('r'),
                   help='input text file')
    parser.add_argument('output', type=argparse.FileType('w'),
                   help='output text file with lower case')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = getArgs()
    
    fout = open(args.output.name, args.output.mode)
    for line in open(args.input.name, args.input.mode):
        fout.write(line.lower())
    
    fout.close()
        