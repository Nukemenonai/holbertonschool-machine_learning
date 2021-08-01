#!/usr/bin/env python3
""" schools by topic """


def schools_by_topic(mongo_collection, topic):
    """ 
    returns the list of schools 
    that have a specific topic"""


    documents = [item for item in mongo_collection.find()]
    doc_filter = []


    for item in documents:
        if 'topics' in item.keys():
            if topic in item['topics']:
                doc_filter.append(item)

    return doc_filter