#!/usr/bin/python
# -*- coding: utf8 -*-

'''
Scrapes Apple Directory and determines doppelgängers
'''

import argparse
import base64
import collections
import heapq
import os
import random
import tempfile
import time

import dlib
import numpy
from testlogger import logger

from . import (
    db,
    ldap_utils,
)


Pipeline = collections.namedtuple('Pipeline', [
    'face_detector',
    'pose_analyzer',
    'face_encoder',
])


def save_bytes_to_file(byte_array):
    '''
    Given an array of bytes, returns the filename of a
    a temporary file that contains those contents
    '''
    handle = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    handle.write(byte_array)
    handle.flush()
    return handle.name


def get_pipeline():
    '''
    Returns the machine for each stage of the machine learning pipeline
    '''

    # Since dlib is a C-linked library, the functions aren't
    # defined in python and pylint hates that, so we ignore them
    # pylint: disable=no-member

    logger.info('Loading face detector')
    face_detector = dlib.get_frontal_face_detector()

    logger.info('Loading pose analyzer')
    pose_bits = os.path.expanduser(
        '~/Downloads/shape_predictor_68_face_landmarks.dat'
    )
    pose_analyzer = dlib.shape_predictor(pose_bits)

    logger.info('Loading face encoder')
    encoder_bits = os.path.expanduser(
        '~/Downloads/dlib_face_recognition_resnet_model_v1.dat'
    )
    face_encoder = dlib.face_recognition_model_v1(encoder_bits)

    return Pipeline(face_detector, pose_analyzer, face_encoder)


def calculate_encoding_for_face(file_name, pipeline):
    '''
    Given the path to some image, calculate the encoding for the faces.
    '''
    face_image = dlib.load_rgb_image(file_name)  # pylint: disable=no-member

    face_locations = pipeline.face_detector(face_image, 1)
    if len(face_locations) != 1:
        logger.warn('Found %s faces', len(face_locations))
        return None

    face_location = face_locations[0]

    pose = pipeline.pose_analyzer(face_image, face_location)

    encoding = pipeline.face_encoder.compute_face_descriptor(
        face_image,
        pose,
        1,
    )

    # Since the face_encoder is from dlib, we actually need to wrap that in a
    # numpy.array() because that's more ergonomic and has better functions to
    # operate with.  We need functions that can save to disk, for example.
    return numpy.array(encoding)


DB_PATH = './doppelganger.db'


def get_database():
    '''
    Returns a connection to the database
    '''
    logger.info('Connecting to database: %s', DB_PATH)
    return db.Database(DB_PATH)


def init(_):
    '''
    Sets up a database for you
    '''
    # All these are expensive to do so we do them once out here
    database = get_database()  # Connections are expensive and limited
    ldap_instance = ldap_utils.init_ldap()  # This is actually over the network
    pipeline = get_pipeline()  # This loads a lot of files and is slow

    for employee in ldap_utils.get_employees(ldap_instance):
        logger.info('Looking at %s', employee['cn'])
        image_bytes = base64.b64decode(employee['applePhotoOfficial-jpeg'])
        file_name = save_bytes_to_file(image_bytes)
        encoding = calculate_encoding_for_face(file_name, pipeline)
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
