'''
Defines the flask web service part of this repository (as opposed to the CLI)
'''

import base64
import collections
import json

from flask import (
    Flask,
    url_for,
    redirect,
    request,
)

from . import (
    ml,
    db,
    logic,
)


APP = Flask(__name__, static_url_path='')


CACHE = {}


@APP.route('/')
def index():
    '''
    Our home page!
    '''
    return redirect(url_for('static', filename='index.html'))


Response = collections.namedtuple('Response', {
    'location',
    'landmarks',
    'twins',
})


@APP.route('/process', methods=['POST'])
def process():
    '''
    Our home page!
    '''
    image_uri = request.form['image_uri']
    _data_header, encoded_data = image_uri.split(',')
    image_bytes = base64.b64decode(encoded_data)

    file_name = ml.save_bytes_to_file(image_bytes)
    pipeline_results = ml.calculate_encoding_for_face(file_name, get_pipeline())

    responses = []
    for pipeline_result in pipeline_results:
        twins = logic.compare(pipeline_result.encoding, get_employees(), 20)
        response = Response(
            location=pipeline_result.location,
            landmarks=pipeline_result.landmarks,
            twins=twins,
        )
        logic.print_twins(twins)
        responses.append(response)

    return json.dumps(responses)


def get_pipeline():
    '''
    Returns the ML pipeline as cached by the flask webapp

    This allows us to not build the pipeline for each request, but just once
    '''
    if 'pipeline' not in CACHE:
        CACHE['pipeline'] = ml.get_pipeline()
    return CACHE['pipeline']


def get_employees():
    '''
    Returns the full list of employee data in memory

    This allows us to not build this list from disk every request
    '''
    if 'employees' not in CACHE:
        CACHE['employees'] = db.Database(db.DB_PATH).get_all()
    return CACHE['employees']
