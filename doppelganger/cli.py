'''
Code mostly used by the CLI version of this app
'''

import argparse
import base64

from testlogger import logger

from . import (
    db,
    ldap_utils,
    ml,
    logic,
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
        logger.info('Looking at %s, %s', employee['cn'], employee['appledsId'])
        image_bytes = base64.b64decode(employee['applePhotoOfficial-jpeg'])
        file_name = ml.save_bytes_to_file(image_bytes)
        results = ml.calculate_encoding_for_face(file_name, pipeline)
        if not results:
            logger.warning('No faces found')
        else:
            if len(results) > 1:
                logger.warning(
                    'Found %s faces, using first', len(results))

            result = results[0]
            entry = db.create_entry_from_record(
                employee,
                result.encoding,
            )
            database.put(entry)


def analyze(args):
    '''
    Given the dsid of a target, find the top matches to that target
    '''
    database = get_database()
    employee = database.get_by_dsid(args.dsid)
    logger.info('Finding matches for %s', employee.name)

    logger.info('Loading all employees')
    all_employees = database.get_all()

    twins = logic.compare(employee.facial_encoding, all_employees, 20)
    logic.print_twins(twins)


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
