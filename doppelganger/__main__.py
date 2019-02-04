'''
This is the main module.  It is executed when run with `python -m doppleganger`
'''

import base64
import json
import os

import cv2

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


def get_cascade_bits():
    '''
    The CascadeClassifier needs some bits from being trained on faces
    '''
    return os.path.join(
        cv2.data.haarcascades,
        'haarcascade_frontalface_default.xml',
    )


def detect_faces(file_name):
    '''
    Given file path, shows the image with faces highlighted
    '''
    classifier = cv2.CascadeClassifier(get_cascade_bits())
    image = cv2.imread(file_name)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(
        grayscale_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),

        # This is no longer the old one based on the following:
        # https://github.com/opencv/opencv/commit/c371df4aa2670d97d08217b351c946a9a8bef09f
        # https://stackoverflow.com/questions/36242860/attribute-error-while-using-opencv-for-face-recognition
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    color = (0, 255, 0)
    stroke_width = 2
    for (x_pos, y_pos, width, height) in faces:
        cv2.rectangle(
            image,
            (x_pos, y_pos),
            (x_pos + width, y_pos + height),
            color,
            stroke_width,
        )

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)


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
        detect_faces(file_name)


if __name__ == '__main__':
    main()
