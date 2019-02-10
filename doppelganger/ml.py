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


def calculate_encoding_for_image(file_name, pipeline):
    '''
    Given the path to some image, calculate the encoding for the faces.
    '''
    face_image = dlib.load_rgb_image(file_name)  # pylint: disable=no-member

    face_locations = pipeline.face_detector(face_image, 1)

    result = []
    for location in face_locations:
        result.append(calculate_encoding_for_face(
            pipeline,
            face_image,
            location,
        ))

    logger.info('Found %s faces', len(result))

    return result


def calculate_encoding_for_face(pipeline, face_image, location):
    '''
    Given the location of a single face to focus on, returns the
    PipelineResult object containing the landmarks and encodings
    for that one specific face.
    '''
    landmarks = pipeline.pose_analyzer(face_image, location)
    encoding = pipeline.face_encoder.compute_face_descriptor(
        face_image,
        landmarks,
        1,  # Upsample once to make things bigger and detect more faces
    )

    # Convert the dlib based answers into python primitives
    # This allows us to do future stuff like json dumps
    encoding = primitivize_encoding(encoding)
    landmarks = primitivize_landmarks(landmarks)
    location = primitivize_location(location)

    return PipelineResult(
        location=location,
        landmarks=landmarks,
        encoding=encoding,
    )


def primitivize_location(location):
    '''
    Given a location from dlib, which is a dlib.rectangle, return a dictionary
    containing similar information.  Only use width, height, x, and y which is
    what rendering APIs in html generally use.  This allows us to transmit
    over the wire as json, for example.
    '''
    width = location.width()
    height = location.height()
    as_primitive = point_to_dict(location.tl_corner())
    as_primitive['width'] = width
    as_primitive['height'] = height
    return as_primitive


def primitivize_landmarks(landmarks):
    '''
    Given landmarks from dlib, which is a dlib.vector of dlib.points, return a
    dictionary list of dictionaries that contain an x and a y (x, y).  This
    makes things json-able.
    '''
    return map(point_to_dict, landmarks.parts())


def primitivize_encoding(encoding):
    '''
    Since the face_encoder is from dlib, we actually need to wrap that
    in a numpy.array() because that's more ergonomic and has better
    functions to operate with.  We need functions that can save to
    disk, for example.  We also need to be able to diff encodings.

    At the time of this writing, encodings are a dlib.array of floats.

    '''
    return numpy.array(encoding)


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
