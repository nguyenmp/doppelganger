'''
Code related to the machine learning bits
'''

import collections
import os
import tempfile

import dlib
import numpy

from testlogger import logger


Pipeline = collections.namedtuple('Pipeline', [
    'face_detector',
    'pose_analyzer',
    'face_encoder',
])


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


def save_bytes_to_file(byte_array):
    '''
    Given an array of bytes, returns the filename of a
    a temporary file that contains those contents
    '''
    handle = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    handle.write(byte_array)
    handle.flush()
    return handle.name
