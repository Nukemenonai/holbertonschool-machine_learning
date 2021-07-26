#!/usr/bin/env python3
""" homeworlds """
import requests


def sentientPlanets():
    """ returns the list of names of the home planets
        of all sentient species.
    """

    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []
    while url is not None:
        page = requests.get(url)
        page = page.json()
        results = page['results']
        for species in results:
            if (species["classification"] == 'sentient' or
                    species["designation"] == 'sentient'):
                if species['homeworld'] is not None:
                    planet = requests.get(species['homeworld'])
                    planets.append(planet.json()['name'])
        url = page["next"]
    return planets