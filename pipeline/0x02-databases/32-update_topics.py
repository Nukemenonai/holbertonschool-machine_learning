#!/usr/bin/env python3
""" update topics"""


def update_topics(mongo_collection, name, topics):
    """changes all topics of a school 
    docuent based on the name"""
    mongo_collection.update(
        {
            'name': name
        },
        {
            '$set': 
            {
                'topics': topics
            }
        }
    )