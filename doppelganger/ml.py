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


PipelineResult = collections.namedtuple('PipelineResult', [
    'location',  # The rectangle around where the face is
    'landmarks',  # The positions of the basic facial landmarks on the image
    'encoding',  # The measurements that differentiate faces
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

    result = []
    for location in face_locations:
        landmarks = pipeline.pose_analyzer(face_image, location)
        encoding = pipeline.face_encoder.compute_face_descriptor(
            face_image,
            landmarks,
            1,  # Upsample once to make things bigger and detect more faces
        )

        # Since the face_encoder is from dlib, we actually need to wrap that
        # in a numpy.array() because that's more ergonomic and has better
        # functions to operate with.  We need functions that can save to
        # disk, for example.  We also need to be able to diff encodings.
        encoding = numpy.array(encoding)

        # the landmarks on the other hand is dlib-class that exports into
        # a list of points so we just keep it that way
        landmarks = map(point_to_dict, landmarks.parts())

        # I don't really have a good justification for why we do this
        # other htan we don't want to pass around a dlib object through a
        # web request.  Other htan that, we can pick any good primitive
        # encoding desired.
        width = location.width()
        height = location.height()
        location = point_to_dict(location.tl_corner())
        location['width'] = width
        location['height'] = height

        result.append(PipelineResult(
            location=location,
            landmarks=landmarks,
            encoding=encoding,
        ))

    logger.info('Found %s faces', len(result))

    return result


def point_to_dict(point):
    '''
    Given a dlib.Point, returns a dictionary with x and y defined
    '''
    return {'x': point.x, 'y': point.y}


def save_bytes_to_file(byte_array):
    '''
    Given an array of bytes, returns the filename of a
    a temporary file that contains those contents
    '''
    handle = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    handle.write(byte_array)
    handle.flush()
    return handle.name
