'''
Tests functions defined in the ml module
'''

from doppelganger import ml

from mock import (
    MagicMock,
    patch,
    call,
)


@patch('doppelganger.ml.dlib.face_recognition_model_v1')
@patch('doppelganger.ml.dlib.shape_predictor')
@patch('doppelganger.ml.dlib.get_frontal_face_detector')
def test_get_pipeline(location_func, landmarks_func, encoding_func):
    '''
    Makes sure the bits we expect are in the right place
    '''
    location_func.return_value = MagicMock()
    landmarks_func.return_value = MagicMock()
    encoding_func.return_value = MagicMock()

    pipeline = ml.get_pipeline()

    location_func.assert_called_once()
    landmarks_func.assert_called_once()
    encoding_func.assert_called_once()

    assert pipeline.face_detector == location_func.return_value
    assert pipeline.pose_analyzer == landmarks_func.return_value
    assert pipeline.face_encoder == encoding_func.return_value


def test_pipeline_result():
    '''
    Since we pass this into a json blob as an array, the ordering is actually
    super important, thus this tests that the order in the tuple is as expected

    This is basically an API compatiblity check.  A change in the size or
    ordering here results in a change in the API consumptions.
    '''
    pipeline = ml.PipelineResult(
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    assert pipeline[0] == pipeline.location
    assert pipeline[1] == pipeline.landmarks
    assert pipeline[2] == pipeline.encoding


@patch('doppelganger.ml.calculate_encoding_for_face')
@patch('doppelganger.ml.dlib.load_rgb_image')
def test_calculate_encoding_image(loader_func, calculator_func):
    '''
    Verifies calculate_encoding_for_image behaves correctly
    '''

    # Mock the inputs
    file_name = MagicMock()
    pipeline = MagicMock()

    # Mock the internal calls
    face_image = MagicMock()
    loader_func.return_value = face_image

    face_locations = [MagicMock(), MagicMock()]
    pipeline.face_detector = MagicMock(return_value=face_locations)
    pipeline_results = [MagicMock(), MagicMock()]
    calculator_func.side_effect = pipeline_results

    # Execute
    result = ml.calculate_encoding_for_image(file_name, pipeline)
    assert result == pipeline_results

    # Check the internal calls
    loader_func.assert_called_once_with(file_name)
    pipeline.face_detector.assert_called_once_with(loader_func.return_value, 1)
    calculator_func.assert_has_calls(
        [call(pipeline, face_image, location) for location in face_locations],
        any_order=True,
    )


@patch('doppelganger.ml.primitivize_encoding')
@patch('doppelganger.ml.primitivize_landmarks')
@patch('doppelganger.ml.primitivize_location')
def test_calculate_encoding_face(location_func, landmarks_func, encoding_func):
    '''
    Checks calculate_encoding_for_face runs properly
    '''
    # Mock inputs
    pipeline = MagicMock()
    face_image = MagicMock()
    location = MagicMock()

    # Mock internals
    landmarks = MagicMock()
    pipeline.pose_analyzer.return_value = landmarks

    encoding = MagicMock()
    pipeline.face_encoder.compute_face_descriptor.return_value = encoding

    primitive_location = MagicMock()
    location_func.return_value = primitive_location

    primitive_landmarks = MagicMock()
    landmarks_func.return_value = primitive_landmarks

    primitive_encoding = MagicMock()
    encoding_func.return_value = primitive_encoding

    # Drive function and check result
    result = ml.calculate_encoding_for_face(pipeline, face_image, location)
    assert result.location == primitive_location
    assert result.landmarks == primitive_landmarks
    assert result.encoding == primitive_encoding

    # Check internals
    pipeline.pose_analyzer.assert_called_once_with(face_image, location)
    pipeline.face_encoder.compute_face_descriptor.assert_called_once_with(
        face_image,
        landmarks,
        1,
    )
    location_func.assert_called_once_with(location)
    landmarks_func.assert_called_once_with(landmarks)
    encoding_func.assert_called_once_with(encoding)


@patch('doppelganger.ml.point_to_dict')
def test_primitivize_location(to_dict_func):
    '''
    Tests primitivize_location's basic flow
    '''
    location = MagicMock()
    location.width.return_value = MagicMock()
    location.height.return_value = MagicMock()
    location.tl_corner.return_value = MagicMock()

    to_dict_func.return_value = {'x': 1, 'y': 2}

    result = ml.primitivize_location(location)
    assert result['width'] == location.width.return_value
    assert result['height'] == location.height.return_value
    assert result['x'] == 1
    assert result['y'] == 2

    to_dict_func.assert_called_once_with(location.tl_corner.return_value)
    location.width.assert_called_once()
    location.height.assert_called_once()


@patch('doppelganger.ml.point_to_dict')
def test_primitivize_landmarks(to_dict_func):
    '''
    Tests primitivize_landmarks's basic flow
    '''
    landmarks = MagicMock()
    points = [MagicMock(), MagicMock()]
    landmarks.parts.return_value = points
    to_dict_result = {'x': 1, 'y': 2}
    to_dict_func.return_value = to_dict_result

    result = ml.primitivize_landmarks(landmarks)
    assert result == [to_dict_result, to_dict_result]

    landmarks.parts.assert_called_once_with()
    to_dict_func.assert_has_calls([call(point) for point in points])


@patch('doppelganger.ml.numpy.array')
def test_primitivize_encoding(array_func):
    '''
    Tests primitivize_encoding's basic flow
    '''
    expected_result = MagicMock()
    array_func.return_value = expected_result
    encoding = MagicMock()

    actual_result = ml.primitivize_encoding(encoding)
    assert expected_result == actual_result


def test_point_to_dict():
    '''
    Checks that this function is wellformed and behaves expectedly
    '''
    x_pos = 24
    y_pos = 80
    import dlib
    point = dlib.point(x_pos, y_pos)  # pylint: disable=no-member
    point_dict = ml.point_to_dict(point)
    assert point_dict['x'] == x_pos
    assert point_dict['y'] == y_pos


def test_save_bytes_to_file():
    '''
    Saves bytes to a file and checks they were properly saved
    '''
    content = '\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00'
    file_name = ml.save_bytes_to_file(content)
    with open(file_name) as handle:
        assert handle.read() == content
