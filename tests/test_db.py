'''
Tests the code in db.py
'''

from doppelganger import db

from mock import (
    patch,
    MagicMock,
)


@patch('doppelganger.db.base64.b64decode')
def test_create_entry_from_record(b64decode_func):
    '''
    Checks that we properly set values in create_entry_from_record
    '''
    record = {
        'appledsId': MagicMock(),
        'cn': MagicMock(),
        'applePhotoOfficial-jpeg': MagicMock(),
    }
    facial_encoding = MagicMock()

    raw_bytes = MagicMock()
    b64decode_func.return_value = raw_bytes

    result = db.create_entry_from_record(record, facial_encoding)
    assert result.dsid == record['appledsId']
    assert result.name == record['cn']
    assert result.facial_encoding == facial_encoding
    assert result.picture == raw_bytes

    b64decode_func.assert_called_once_with(record['applePhotoOfficial-jpeg'])


@patch('doppelganger.db.str')
@patch('doppelganger.db.bin_to_nparray')
def test_create_entry_from_row(bin_to_nparray_func, str_func):
    '''
    Checks that we properly set values in create_entry_from_row
    '''
    row = {
        'dsid': MagicMock(),
        'name': MagicMock(),
        'facial_encoding': MagicMock(),
        'picture': MagicMock(),
    }

    np_array = MagicMock()
    bin_to_nparray_func.return_value = np_array

    str_result = MagicMock()
    str_func.return_value = str_result

    result = db.create_entry_from_row(row)
    assert result.dsid == row['dsid']
    assert result.name == row['name']
    assert result.facial_encoding == np_array
    assert result.picture == str_result

    str_func.assert_called_once_with(row['picture'])
    bin_to_nparray_func.assert_called_once_with(row['facial_encoding'])


def test_numpy_array_serialization():
    '''
    Tests serialization and deserialization of numpy arrays
    '''
    import numpy
    array = numpy.array([0, 1, 2])
    binary = db.nparray_to_bin(array)
    assert binary
    and_back_again = db.bin_to_nparray(binary)
    assert numpy.array_equal(array, and_back_again)
