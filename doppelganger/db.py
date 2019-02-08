'''
All the code pertaining to the file database backing
'''

import base64
import collections
import io
import sqlite3
import tempfile

import numpy


Entry = collections.namedtuple('Entry', [
    'name',  # A utf8 encoded string
    'dsid',  # A string?
    'facial_encoding',  # A numpy array
    'picture',  # The binary blob of jpeg bits
])

INIT_SCRIPT = '''
CREATE TABLE IF NOT EXISTS entry (
    dsid INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    facial_encoding BLOB NOT NULL,
    picture BLOB NOT NULL
);
'''


def create_entry_from_record(record, facial_encoding):
    '''
    Given some record from active directory, returns an Entry
    '''
    return Entry(
        dsid=record['appledsId'],
        name=record['cn'],
        facial_encoding=facial_encoding,
        picture=base64.b64decode(record['applePhotoOfficial-jpeg']),
    )


def create_entry_from_row(row):
    '''
    Converts a row in the DB into an Entry object
    '''
    return Entry(
        dsid=row['dsid'],
        name=row['name'],
        facial_encoding=bin_to_nparray(row['facial_encoding']),
        picture=row['picture'],
    )


class Database(object):
    '''
    Encapsulates the interaction with a single database
    '''

    def __init__(self, path):
        '''
        Establishes a connection to a database defined by path
        '''
        self.connection = sqlite3.connect(path)

        # allow us to access row values by string
        self.connection.row_factory = sqlite3.Row

        self.connection.executescript(INIT_SCRIPT)

    def entries(self):
        '''
        Generator for entries in the DB
        '''
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM entry')
        for row in cursor:
            yield create_entry_from_row(row)

    def get_by_dsid(self, dsid):
        '''
        Given a DSID, return the exact match employee Entry from our database
        '''
        statement = '''
            SELECT dsid, name, facial_encoding, picture
            FROM entry
            WHERE dsid=?
        '''
        cursor = self.connection.cursor()
        cursor.execute(statement, (dsid,))
        row = cursor.fetchone()
        return create_entry_from_row(row)

    def put(self, entry):
        '''
        Given an Entry tuple, insert this into the database,
        overwriting any prior matching entry by DSID
        '''
        cursor = self.connection.cursor()
        statement = 'INSERT OR REPLACE INTO entry VALUES(?, ?, ?, ?)'
        values = (
            entry.dsid,
            entry.name,
            sqlite3.Binary(nparray_to_bin(entry.facial_encoding)),
            sqlite3.Binary(entry.picture),
        )
        cursor.execute(statement, values)
        self.connection.commit()


def nparray_to_bin(nparray):
    '''
    Converts a numpy array into some binary data that can be stored

    This only exists because numpy only provides binary conversion using files
    '''
    # Creates a file like object in memory to write the bytes to temporarily
    file_ish = io.BytesIO()
    numpy.save(file_ish, nparray)
    return file_ish.getvalue()


def bin_to_nparray(binary):
    '''
    Given some binary data that was once a numpy array, read it back

    This only exists because numpy only provides binary conversion using files
    '''
    # Converts the bits into a file like object
    file_ish = io.BytesIO(binary)
    return numpy.load(file_ish)
