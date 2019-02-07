'''
This is the main module.  It is executed when run with `python -m doppleganger`
'''

from . import (
    logger,
    argument_parser,
)


def main():
    '''
    Finds all employees and their profile pictures
    '''
    logger.info('Starting process')
    args = argument_parser().parse_args()
    args.func()


if __name__ == '__main__':
    main()
