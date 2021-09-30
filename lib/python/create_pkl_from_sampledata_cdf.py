#!/usr/bin/env python3

import sys
from sampledata import load_sampledata

for i in range(1,len(sys.argv)):
    print("Processing input file: " + sys.argv[i])
    load_sampledata(sys.argv[i])
print("Done")
