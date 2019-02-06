#!/usr/bin/python
# -*- coding: utf8 -*-

'''
Scrapes Apple Directory and determines doppelgängers
'''

import base64
import tempfile

import ldap
from ldap import asyncsearch
import numpy
from testlogger import logger


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


def process_result(result_datas):
    '''
    Given an ldap result, returns a normal looking dictionary pulling out
    all the values from their wrapping arrays and stuff.
    '''

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

    image_data = attributes['applePhotoOfficial-jpeg']
    attributes['applePhotoOfficial-jpeg'] = base64.b64encode(image_data)
    return attributes
