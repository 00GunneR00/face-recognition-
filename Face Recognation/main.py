import cv2
import face_recognition
from KisiEkleme import KisiEkleme


if __name__ == "__main__":
    yuz_tanima = {
        'image1': 'image1.jpg',
        'image2': 'image2.jpg',
        'image3': 'image3.jpg',
        'image4': 'image4.jpg',
    }

    app = KisiEkleme(yuz_tanima)
    app.run()