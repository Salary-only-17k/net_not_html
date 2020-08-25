import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
from time import sleep
import numpy as np
import dlib
import tensorflow as tf
import time

class age_gender:
    def __init__(self, path, face_size=64):
        self.path = path
        self.face_size = face_size

    @classmethod
    def _draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


    def _rect_to_bb(self,rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)


    def run(self):
        face_cascade = dlib.get_frontal_face_detector()
        video_capture = cv2.VideoCapture(0)
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            with tf.gfile.FastGFile(self.path, 'rb+') as f:
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
            while True:
                if not video_capture.isOpened():
                    sleep(5)
                ret, frame = video_capture.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = face_cascade(gray, 1)
                index_y=[]
                face_imgs = np.empty((len(rects), self.face_size, self.face_size, 3))
                for i, rect in enumerate(rects):
                    (x, y, w, h) = self._rect_to_bb(rect)
                    index_y.append([x,y])
                    face =frame[y + 5:y + h + 5, x + 5:x + w + 5]
                    if w != self.face_size and h != self.face_size:
                        face_img = cv2.resize(face, (self.face_size, self.face_size))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    face_imgs[i, :, :, :] = face_img
                if len(face_imgs) > 0:
                    sess.run(tf.global_variables_initializer())
                    img = sess.graph.get_tensor_by_name("input_1:0")
                    pre_gender = sess.graph.get_tensor_by_name("dense_1/Softmax:0")
                    pre_age = sess.graph.get_tensor_by_name('dense_2/Softmax:0')
                    s = time.time()


                    predicted_genders =sess.run(pre_gender,feed_dict={img:face_imgs})
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = sess.run(pre_age, feed_dict={img: face_imgs}).dot(ages).flatten()
                    print('cost time : ', time.time() - s)

                else:
                    print('没有人脸')
                for i, x_y in enumerate(index_y):

                    label = f"{int(predicted_ages[i])}, {'F'if predicted_genders[i][0] > 0.5 else 'M'}"
                    self._draw_label(frame, (x_y[0],x_y[1]), label)

                cv2.imshow('pre_age_gender', frame)
                if cv2.waitKey(20) == 27:
                    break

            # When everything is done, release the capture
            video_capture.release()
            cv2.destroyAllWindows()


def main():

    path='tmp/age_gender01.pb'
    ag = age_gender(path)

    ag.run()


if __name__ == "__main__":
    main()
