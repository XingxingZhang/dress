#!/usr/bin/python

from readability import Readability
import sys

if __name__ == '__main__':
    infile = sys.argv[1]
    text = open(infile).read()
    rd = Readability(text)
    print(rd.FleschKincaidGradeLevel())

