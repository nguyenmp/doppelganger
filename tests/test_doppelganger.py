'''
Tests for doppelganger
'''

import doppelganger


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
