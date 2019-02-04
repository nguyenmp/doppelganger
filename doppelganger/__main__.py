'''
This is the main module.  It is executed when run with `python -m doppleganger`
'''

import base64
import json
import subprocess

from . import (
    init_ldap,
    get_employees,
    logger,
)


def save_bytes_to_file(byte_array):
    '''
    Given an array of bytes, returns the filename of a
    a temporary file that contains those contents
    '''
    import tempfile
    handle = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
    handle.write(byte_array)
    handle.flush()
    return handle.name


def main():
    '''
    Finds all employees and their profile pictures
    '''
    ldap_instance = init_ldap()
    for employee in get_employees(ldap_instance):
        json.dumps(employee)
        logger.info('Looking at %s', json.dumps(employee))
        image_bytes = base64.b64decode(employee['applePhotoOfficial-jpeg'])
        file_name = save_bytes_to_file(image_bytes)
        subprocess.check_call('open {}'.format(file_name), shell=True)
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
