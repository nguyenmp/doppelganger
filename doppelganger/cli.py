'''
Code mostly used by the CLI version of this app
'''

import argparse
import base64
import heapq
import numpy

from testlogger import logger

from . import (
    db,
    ldap_utils,
    ml,
)


def get_database():
    '''
    Returns a connection to the database
    '''
    logger.info('Connecting to database: %s', db.DB_PATH)
    return db.Database(db.DB_PATH)


def init(_):
    '''
    Sets up a database for you
    '''
    # All these are expensive to do so we do them once out here
    database = get_database()  # Connections are expensive and limited
    ldap_instance = ldap_utils.init_ldap()  # This is actually over the network
    pipeline = ml.get_pipeline()  # This loads a lot of files and is slow

    for employee in ldap_utils.get_employees(ldap_instance):
        logger.info('Looking at %s', employee['cn'])
        image_bytes = base64.b64decode(employee['applePhotoOfficial-jpeg'])
        file_name = ml.save_bytes_to_file(image_bytes)
        encoding = ml.calculate_encoding_for_face(file_name, pipeline)
        if encoding is not None:
            employee['encoding'] = encoding
            entry = db.create_entry_from_record(
                employee,
                encoding,
            )
            database.put(entry)


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
        # We use negative distance becauset his is a min heap
        # and we want to pop the furthest items efficiently
        heapq.heappush(twins, (-distance, employee))

        while len(twins) > count:
            heapq.heappop(twins)

    return twins


def print_twins(twins):
    '''
    Given a list of twins, print them out in a good order
    '''
    twins = sorted(twins, key=lambda x: x[0])
    for (distance, twin) in twins:
        logger.info(
            'DSID: %s matches %s%%',
            twin.dsid,
            int(100 - (-distance * 100)),
        )


def analyze(args):
    '''
    Given the dsid of a target, find the top matches to that target
    '''
    database = get_database()
    employee = database.get_by_dsid(args.dsid)
    logger.info('Finding matches for %s', employee.name)

    logger.info('Loading all employees')
    all_employees = []
    for entry in database.entries():
        all_employees.append(entry)

    import pdb
    pdb.set_trace()

    twins = compare(employee.facial_encoding, all_employees, 20)
    print_twins(twins)


def argument_parser():
    '''
    The processor of arguments
    '''
    logger.info('Building argument parser')
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    init_parser = subparsers.add_parser('init')
    init_parser.set_defaults(func=init)

    init_parser = subparsers.add_parser('analyze')
    init_parser.add_argument('dsid', type=int, help='the person to match with')
    init_parser.add_argument(
        'count', type=int,
        help='the number of matches to retain',
    )
    init_parser.set_defaults(func=analyze)

    return parser
