#!/usr/bin/env python3
""" SWAPI Passengers"""
import requests


def availableShips(passengerCount):
    """Returns the list of ships that 
    can hold a given number of passengers"""
    url = 'https://swapi-api.hbtn.io/api/starships/'
    data = requests.get(url)
    data = data.json()
    my_ships = []

    while(data['next']):
        for result in data['results']:
            passengers = result['passengers']
            passengers = passengers.replace(',', '')
            if passengers.isnumeric():
                if int(passengers) >= passengerCount:
                    my_ships.append(result['name'])
        data = requests.get(data['next'])
        data = data.json()
        
    if data['next'] is None:
        for result in data['results']:
            passengers = result['passengers']
            passngers = passengers.replace(',', '')
            if passengers.isnumeric():
                if int(passengers) >= passengerCount:
                    my_ships.append(result['name'])

    return my_ships
