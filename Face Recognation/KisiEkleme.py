import cv2
import face_recognition

class KisiEkleme:
    def __init__(self, known_faces):
        self.known_faces = known_faces
        self.face_encodings = []
        self.face_names = []
        for name, image_path in known_faces.items():
            img = face_recognition.load_image_file(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(img)[0]
            self.face_encodings.append(encoding)
            self.face_names.append(name)

    def run(self):
        vid = cv2.VideoCapture(0)

        while True:
            ret, frame = vid.read()
            face_locs = face_recognition.face_locations(frame)
            encodings = face_recognition.face_encodings(frame, face_locs)

            for face_loc, encoding in zip(face_locs, encodings):
                matches = face_recognition.compare_faces(self.face_encodings, encoding)
                face_distances = face_recognition.face_distance(self.face_encodings, encoding)
                match_index = matches.index(True) if True in matches else -1

                top, right, bottom, left = face_loc
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

                if match_index != -1:
                    name = self.face_names[match_index]
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                else:
                    cv2.putText(frame, 'Unknown', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()