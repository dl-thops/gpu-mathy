"""
    This python script can be used to generate output (cuda program) files
    for all the input files present in INPUT_DIR.
"""

import os
import subprocess

INPUT_DIR = "benchmark/input/" 
OUTPUT_DIR = "benchmark/output/"
SRC_DIR = "src/"

# Make executable from source files
_ = subprocess.run( 'make -C ' + SRC_DIR, shell = True, stdout=open(os.devnull, 'wb'))

# Run executable on input files
for root, dirs, files in os.walk(INPUT_DIR):
    for name in files:
        output_filename = name.split('.')[0] + '.cu'
        _ = subprocess.run( 'cd ' + SRC_DIR + ' && ./a.out ' + '../' + os.path.join(root, name)\
             + ' ../' + OUTPUT_DIR + output_filename, shell = True )
        print("Generated " + OUTPUT_DIR + output_filename)

_ = subprocess.run( 'make clean -C ' + SRC_DIR, shell = True, stdout=open(os.devnull, 'wb'))
