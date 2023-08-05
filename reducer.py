#!/usr/bin/env python3

import sys

genre_counts = {}

for line in sys.stdin:
    # Remove leading and trailing whitespace
    line = line.strip()
    
    # Split the line into key and value
    genres, count = line.split('\t', 1)
    
    # Convert count to an integer
    try:
        count = int(count)
    except ValueError:
        continue
    
    # Split the genres into individual genres
    individual_genres = genres.split('|')
    
    # Update the count for each individual genre
    for genre in individual_genres:
        if genre in genre_counts:
            genre_counts[genre] += count
        else:
            genre_counts[genre] = count

# Output the genre counts
for genre, count in genre_counts.items():
    print(f"{genre}\t{count}")

