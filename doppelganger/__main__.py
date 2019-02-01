'''
This is the main module.  It is executed when run with `python -m doppleganger`
'''

import json

from . import (
    init_ldap,
    get_employees,
    logger,
)


def main():
    '''
    Finds all employees and their profile pictures
    '''
    ldap_instance = init_ldap()
    for employee in get_employees(ldap_instance):
        json.dumps(employee)
        logger.info('Looking at %s', json.dumps(employee))
        # handle = tempfile.NamedTemporaryFile(mode='w+b', suffix='.jpeg')
        # handle.write(image_data)
        # handle.flush()
        # subprocess.check_call('open {}'.format(handle.name), shell=True)
        # import pdb
        # pdb.set_trace()


if __name__ == '__main__':
    main()
