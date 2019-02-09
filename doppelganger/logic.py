'''
This module contains the business logic around gluing things together
'''

import base64
import collections
import heapq

import numpy
from testlogger import logger


Twin = collections.namedtuple('Twin', [
    'distance',  # Some number between 0 and 1, 1 being far, 0 being identical
    'name',  # unicode name
    'dsid',
    'picture',  # base 64 encoded jpeg
])


def print_twins(twins):
    '''
    Given a list of twins, print them out in a good order
    '''
    for twin in twins:
        logger.info(
            'DSID: %s Name: %s matches %s%%',
            twin.dsid,
            twin.name,
            int(100 - (twin.distance * 100)),
        )


def compare(candidate_facial_encoding, employees, count):
    '''
    Given some target facial encoding, find the `count` most
    similar employees as a tuple of (distance, Entry)
    '''
    logger.info('Comparing')
    twins = []
    for employee in employees:
        distance = numpy.linalg.norm(
            employee.facial_encoding - candidate_facial_encoding
        )

        # Unconditionally add, then remove the later
        # We use negative distance because this is a min heap
        # and we want to pop the furthest items efficiently
        # Thus 0 aka identity would stay zero and be at the top
        # 1 aka opposite would be -1 and would be popped first
        twin = Twin(
            -distance,
            employee.name,
            employee.dsid,
            base64.b64encode(employee.picture),
        )
        heapq.heappush(twins, twin)

        while len(twins) > count:
            heapq.heappop(twins)

    # Before we return this, we should invert the distances
    # again because it's more intuitive as a positive number
    twins = [
        Twin(-twin.distance, twin.name, twin.dsid, twin.picture)
        for twin in twins
    ]

    twins = sorted(twins, key=lambda twin: twin.distance)

    # Return the twins with intuitive distances
    return twins
