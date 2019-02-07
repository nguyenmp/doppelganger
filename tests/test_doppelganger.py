'''
Tests for doppelganger
'''

import doppelganger

from mock import (
    patch,
    MagicMock,
)


EXAMPLE_RESULT = [(
    'appledsId=23151044, ou=People, o=Apple',
    {
        'appledsId': ['23151044'],
        'applePhotoOfficial-jpeg': [
            '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00',
        ],
        'cn': ['Elodia Anguiano Pantoja'],
    }
)]


@patch('doppelganger.init')
def test_arguments_init(init_func):
    '''
    Checks that the argument parser resolves
    the init command to the proper init function
    '''
    parser = doppelganger.argument_parser()
    assert parser.parse_args(['init']).func == init_func


@patch('doppelganger.db.Database')
def test_get_db(database_class):
    '''
    Checks that the argument parser resolves
    the init command to the proper init function
    '''
    database = doppelganger.get_database()
    assert database is not None
    database_class.assert_called_with(doppelganger.DB_PATH)


def test_numpy_array_serialization():
    '''
    Tests serialization and deserialization of numpy arrays
    '''
    import numpy
    array = numpy.array([0, 1, 2])
    binary = doppelganger.nparray_to_bin(array)
    assert binary
    and_back_again = doppelganger.bin_to_nparray(binary)
    assert numpy.array_equal(array, and_back_again)


def test_save_bytes_to_file():
    '''
    Saves bytes to a file and checks they were properly saved
    '''
    content = '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00'
    file_name = doppelganger.save_bytes_to_file(content)
    with open(file_name) as handle:
        assert handle.read() == content


def test_get_filter_string():
    '''
    Asserts that the function call returns the exact string we expect
    '''
    expected = '(&(employeeType=Apple Employee)(applePhotoOfficial-jpeg=*))'
    assert doppelganger.get_filter_string() == expected


def test_process_result():
    '''
    Makes sure that given some standard looking
    result from LDAP, we mutate it as expected
    '''
    result = doppelganger.process_result(EXAMPLE_RESULT)

    # Normally, this key is an array of integer strings, we should pull it out
    assert result['appledsId'] == '23151044'

    # Normally, this is an array of name strings, we should pull it out
    assert result['cn'] == 'Elodia Anguiano Pantoja'

    # Normally, this is a byte-array, we should base64 encode it for usability
    assert result['applePhotoOfficial-jpeg'] == '/9j/4AAQSkZJRgABAQEASABIAAA='


@patch('doppelganger.dlib.face_recognition_model_v1')
@patch('doppelganger.dlib.shape_predictor')
@patch('doppelganger.dlib.get_frontal_face_detector')
def test_get_pipeline(face_detector, pose_analyzer, face_encoder):
    '''
    Checks that we properly construct the pipeline
    '''
    face_detector.return_value = MagicMock()
    pose_analyzer.return_value = MagicMock()
    face_encoder.return_value = MagicMock()

    pipeline = doppelganger.get_pipeline()
    assert pipeline.face_detector == face_detector.return_value
    assert pipeline.pose_analyzer == pose_analyzer.return_value
    assert pipeline.face_encoder == face_encoder.return_value
