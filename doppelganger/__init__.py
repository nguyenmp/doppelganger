#!/usr/bin/python
# -*- coding: utf8 -*-

'''
Scrapes Apple Directory and determines doppelgängers
'''

import argparse
import base64
import collections
import os
import tempfile

import dlib
import ldap
import numpy
from testlogger import logger

from . import (
    db,
)


Pipeline = collections.namedtuple('Pipeline', [
    'face_detector',
    'pose_analyzer',
    'face_encoder',
])


def nparray_to_bin(nparray):
    '''
    Converts a numpy array into some binary data that can be stored

    This only exists because numpy only provides binary conversion using files
    '''
    path = tempfile.NamedTemporaryFile(delete=False).name
    numpy.save(path, nparray)
    with open(path + '.npy', 'rb') as handle:
        return handle.read()


def bin_to_nparray(binary):
    '''
    Given some binary data that was once a numpy array, read it back

    This only exists because numpy only provides binary conversion using files
    '''
    path = tempfile.NamedTemporaryFile(delete=False).name
    with open(path, 'wb') as handle:
        handle.write(binary)

    return numpy.load(path)


def save_bytes_to_file(byte_array):
    '''
    Given an array of bytes, returns the filename of a
    a temporary file that contains those contents
    '''
    handle = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    handle.write(byte_array)
    handle.flush()
    return handle.name


def init_ldap():
    '''
    Initializes a connection to the ldap server
    '''
    ldap_instance = ldap.initialize('ldap://lookup.apple.com')
    ldap_instance.simple_bind_s("", "")
    return ldap_instance


def get_filter_string():
    '''
    Function for generating the filter string to query LDAP with
    '''
    # This contains all the keys and their expected values
    key_values = {
        # Only Apple Employees, no meeting rooms, buildings, contractors
        'employeeType': 'Apple Employee',

        # Only ones with photos that we can look at, many people don't photos
        'applePhotoOfficial-jpeg': '*',
    }

    # Join all those keys and their expected values into an RFC 4515 query
    return "(&{})".format(
        ''.join([
            '({}={})'.format(key, value)
            for (key, value) in key_values.items()
        ])
    )


def get_employees(ldap_instance):
    '''
    This is a generator where, given an LDAP instance, we ask it for all Apple
    Employees with photos, returning a dictionary of name, id, and photo
    '''
    logger.info('Getting employees')
    result_id = ldap_instance.search(
        "o=Apple",
        ldap.SCOPE_SUBTREE,  # pylint: disable=maybe-no-member
        filterstr=get_filter_string(),
        attrlist=['applePhotoOfficial-jpeg', 'cn', 'appledsId'],
    )
    while True:
        # The first value is the result-type which is unused
        (_, result_datas) = ldap_instance.result(
            msgid=result_id,
            all=0,
        )
        yield process_result(result_datas)


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


def process_result(result_datas):
    '''
    Given an ldap result, returns a normal looking dictionary pulling out
    all the values from their wrapping arrays and stuff.
    '''
    logger.info('Processing employee')

    # The docs are ambiguous on twhat result_data is so I'm guessing here
    # and asserting my assumptions.  I think it's always an array of 1
    # because we set all=0 which returns results one at a time.  I guess if
    # I passed all=1, then result_data would be a list of all results.
    assert len(result_datas) == 1
    result_data = result_datas[0]

    # The first value is the distinguished name (dn), we don't use it
    (_, attributes) = result_data

    # The attributes are stored as an array of values.  We just want to
    # pull it out, but it's rather undocumented why it's stored this way so
    # I added an assert in case this assumption is wrong
    for (key, value) in attributes.items():
        assert len(value) == 1
        assert isinstance(value, list)
        attributes[key] = value[0]

    attributes['cn'] = attributes['cn'].decode('utf8')
    logger.info('Found %s', attributes['cn'])

    image_data = attributes['applePhotoOfficial-jpeg']
    attributes['applePhotoOfficial-jpeg'] = base64.b64encode(image_data)
    return attributes


DB_PATH = './doppelganger.db'


def get_database():
    '''
    Returns a connection to the database
    '''
    logger.info('Connecting to database: %s', DB_PATH)
    return db.Database(DB_PATH)


def init():
    '''
    Sets up a database for you
    '''
    # All these are expensive to do so we do them once out here
    database = get_database()  # Connections are expensive and limited
    ldap_instance = init_ldap()  # This is actually over the network
    pipeline = get_pipeline()  # This loads a lot of files and is slow

    for employee in get_employees(ldap_instance):
        logger.info('Looking at %s', employee['cn'])
        image_bytes = base64.b64decode(employee['applePhotoOfficial-jpeg'])
        file_name = save_bytes_to_file(image_bytes)
        encoding = calculate_encoding_for_face(file_name, pipeline)
        if encoding is not None:
            employee['encoding'] = encoding
            entry = db.create_entry_from_record(
                employee,
                nparray_to_bin(encoding),
            )
            database.put(entry)


def argument_parser():
    '''
    The processor of arguments
    '''
    logger.info('Building argument parser')
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    init_parser = subparsers.add_parser('init')
    init_parser.set_defaults(func=init)

    return parser
