#!/usr/bin/env python3
""" 
Github API 
user location 
"""
import requests
import sys


if __name__ == '__main__':
    """prints the location of a specific user
    """
    url = sys.argv[1]
    headers = {'Accept': 'application/vnd.github.v3+json'}
    data = requests.get(url, headers=headers)

    if data.status_code == 200:
        print(data.json()['location'])

    if data.status_code == 404:
        print("Not found")

    if data.status_code == 403:
        limit = int(data.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        result = int((limit - now) / 60)
        print("Reset in {} min".format(result))