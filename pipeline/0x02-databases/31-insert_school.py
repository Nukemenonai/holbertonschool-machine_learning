#!/usr/bin/env python3
""" insert document """ 


def insert_school(mongo_collection, **kwargs):
    """ inserts a new documents in a collection """

    mongo_collection.insert(kwargs)
    new_elem = mongo_collection.find(kwargs)
    return new_elem.__dict__['_Cursor__spec']['_id']