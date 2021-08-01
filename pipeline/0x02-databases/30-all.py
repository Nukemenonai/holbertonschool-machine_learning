#!/usr/bin/env python3
""" list all """

def list_all(mongo_collection):
    """ lists all documents in a collection"""

    documents = [elem for elem in mongo_collection.find()]
    return documents