#!/usr/bin/env python3

import sys

# Read the first line to skip the header
header = sys.stdin.readline()

for line in sys.stdin:
    # Remove leading and trailing whitespace
    line = line.strip()
    
    # Split the line into fields
    fields = line.split(',')
    
    # Check if the line has the expected number of fields
    if len(fields) == 5:
        # Extract the genres from the fields
        genres = fields[4].split('|')
        
        # Loop through each genre and output it with a count of 1
        for genre in genres:
            print(f"{genre}\t1")
