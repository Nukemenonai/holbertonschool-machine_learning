#!/usr/bin/env python3
"""
Question answer loop 
"""

while True:

    e = ["exit", "quit", "goodbye", "bye"]

    entry = input("Q: ") 

    if entry.lower() in e:
        print("A: Goodbye")
        break
    else:
        print("A: ")