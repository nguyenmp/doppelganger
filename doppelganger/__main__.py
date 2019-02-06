'''
This is the main module.  It is executed when run with `python -m doppleganger`
'''

import base64
import os

import cv2
import numpy

from . import (
    init_ldap,
    get_employees,
    logger,
    save_bytes_to_file,
)


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
    assert len(faces) == 1
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


def align_face(file_name):
    import dlib
    from imutils.face_utils import (
        FaceAligner,
        rect_to_bb,
    )
    import imutils
    face_detector = dlib.get_frontal_face_detector()
    predictor_bits = '/Users/livingon/Downloads/face-alignment/shape_predictor_68_face_landmarks.dat'
    face_pose_predictor = dlib.shape_predictor(predictor_bits)
    face_aligner = FaceAligner(face_pose_predictor, desiredFaceWidth=256)

    image = cv2.imread(file_name)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = face_detector(gray, 2)

    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = face_aligner.align(image, gray, rect)

        cv2.imshow("Original", faceOrig)
        cv2.imshow("Aligned", faceAligned)
        cv2.waitKey(0)

import dlib
face_detector = dlib.get_frontal_face_detector()
pose_predictor = dlib.shape_predictor('/Users/livingon/Downloads/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('/Users/livingon/Downloads/dlib_face_recognition_resnet_model_v1.dat')

def calculate_encoding_for_face(file_name):
    # Models Loaded
    face_image = cv2.imread(file_name)
    face_locations = face_detector(face_image, 1)
    if len(face_locations) != 1:
        logger.warn('Found %s faces', len(face_locations))
        return None
    face_location = face_locations[0]
    pose = pose_predictor(face_image, face_location)
    return numpy.array(face_encoder.compute_face_descriptor(face_image, pose, 1))


def main():
    '''
    Finds all employees and their profile pictures
    '''
    ldap_instance = init_ldap()
    employees = []
    for employee in get_employees(ldap_instance):
        logger.info('Looking at %s', employee['cn'])
        image_bytes = base64.b64decode(employee['applePhotoOfficial-jpeg'])
        file_name = save_bytes_to_file(image_bytes)
        # detect_faces(file_name)
        # align_face(file_name)
        encoding = calculate_encoding_for_face(file_name)
        if encoding is not None:
            employee['encoding'] = encoding
            employees.append(employee)

    employees = sorted(employees, key=lambda x: numpy.linalg.norm(x['encoding'] - employees[39]['encoding']))
    for employee in employees:
        image_bytes = base64.b64decode(employee['applePhotoOfficial-jpeg'])
        file_name = save_bytes_to_file(image_bytes)
        detect_faces(file_name)


if __name__ == '__main__':
    main()
