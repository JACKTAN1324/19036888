#!/usr/bin/env python3

import sys

genre_counts = {}

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
        
        # Loop through each genre and update the counts
        for genre in genres:
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1

# Output the genre counts
for genre, count in genre_counts.items():
    print(f"{genre}\t{count}")
