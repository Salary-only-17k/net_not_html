"""
This is the code behind the Switching Eds blog post:
    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
See the above for an explanation of the code below.
To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:
    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:
    ./faceswap.py <head image> <face image>
If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.
"""

import cv2
import dlib
import numpy
import os
import sys
from tqdm import tqdm
import shutil

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1  #
FEATHER_AMOUNT = 11

# 关键点的序号,这里可以修改,改成81 或者 5个关键点
LEFT_FACE_POINTS = list([0, 2, 4, 6])
RIGHT_FACE_POINTS = list(range(10, 16))
RIGHT_EYEBROW_POINTS = [22, 24, 26]
LEFT_EYEBROW_POINTS = [17, 19, 21]
NOSE_POINTS = [27]

UP_FACE_POINTS = RIGHT_EYEBROW_POINTS + LEFT_EYEBROW_POINTS
DOTTON_FACE_POINTS = list(range(5, 11))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_FACE_POINTS + RIGHT_FACE_POINTS + UP_FACE_POINTS + DOTTON_FACE_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
# 所有的人脸关键点的编号
OVERLAY_POINTS = [LEFT_FACE_POINTS + RIGHT_FACE_POINTS + UP_FACE_POINTS + DOTTON_FACE_POINTS]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()  # 脸部识别器
predictor = dlib.shape_predictor(PREDICTOR_PATH)  # 导入68点模型数据


def some_cal(landmarks, b_size):
    def conv2int(a, b):
        c = a * b
        return int(c)

    cha1 = landmarks[21][0] - landmarks[17][0]
    cha2 = landmarks[26][0] - landmarks[22][0]
    if cha1 > cha2:
        A = landmarks[LEFT_EYEBROW_POINTS]
        B = landmarks[LEFT_FACE_POINTS]
    else:
        A = landmarks[RIGHT_EYEBROW_POINTS]
        B = landmarks[RIGHT_FACE_POINTS]
    NOSE = landmarks[NOSE_POINTS]
    ratew = [(A[0] - A[1]) / (A[0] - NOSE[0]), (A[0] - A[2]) / (A[0] - NOSE[0]), (A[0] - NOSE[0]) / (A[0] - NOSE[0])]
    w, h, _ = b_size
    bw = int(w / 2) - 40
    w_index = [conv2int(ratew[0], bw) + 40, conv2int(ratew[1]+ratew[0], bw) + 40, conv2int(ratew[2], bw) + 40,
               conv2int(ratew[0], bw) + int(w / 2), conv2int(ratew[1], bw) + int(w / 2),
               conv2int(ratew[2], bw) + int(w / 2)]
    bh = int(h / 2) - 40
    rateh = [(B[0] - B[1]) / (B[0] - B[3]), (B[1] - B[2]) / (B[0] - NOSE[3])]
    h_index = [conv2int(bh, rateh[0])+20,]


def _get_landmarks(im):
    # 判断图中是否只有一个脸
    rects = detector(im, 1)

    return numpy.mat([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def _draw_convex_hull(im, points, color):
    # 圈出人脸轮廓
    points = cv2.convexHull(points)  # 凸包
    cv2.fillConvexPoly(im, points, color=color)  # 绘制填充的多边形


def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        _draw_convex_hull(im,
                          landmarks[group],
                          color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    普式变换
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    """
    读取图片 并且调整大小
    :param fname:  图片
    :return: 调整大小后的图片   关键点坐标
    """
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = _get_landmarks(im)
    # print("s  value : ", s.shape)  # s  value :  (68, 2)
    return im, s


def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)  # 仿射变换
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))


if __name__ == "__main__":
    # 目标
    b_size = [550, 250, 3]
    backgroup = numpy.ones(b_size)
    # im1, landmarks1 = read_im_and_landmarks(r"face_img/mengnalisha.jpg")
    p = 'video/out'
    log = open('log.txt', 'w')
    imgs_paths = [os.path.join(p, i) for i in os.listdir(p)]
    print('开始换脸...')
    for path in tqdm(imgs_paths):
        # 表情
        try:
            im2, landmarks2 = read_im_and_landmarks(path)
            some_cal(backgroup, landmarks2, b_size)
            M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                           landmarks2[ALIGN_POINTS])

            mask = get_face_mask(im2, landmarks2)
            warped_mask = warp_im(mask, M, im1.shape)
            combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                                      axis=0)

            warped_im2 = warp_im(im2, M, im1.shape)
            warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

            output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
            # output_im = output_im.astype(numpy.int8)
            # print(output_im.astype(numpy.int))
            # cv2.imshow("asd",output_im)
            name = os.path.basename(path)
            cv2.imwrite('video/reout/re' + name, output_im)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        except:
            log.write(path + '\n')
            shutil.move(path, 'video/gc')
            imgs_paths.remove(path)
            continue
    log.close()
    print('开始拼接图片 原始图片| 改后图片')
    p = 'video/reout'
    re_imgs_paths = [os.path.join(p, i) for i in os.listdir(p)]
    for i, j in tqdm(zip(imgs_paths, re_imgs_paths)):
        img = cv2.imread(i)
        img = img[100:644, 0:544]
        img = cv2.resize(img, (440, 390))
        reimg = cv2.imread(j)
        pinjie = numpy.hstack([img, reimg])
        cv2.imwrite('video/pinjie/' + os.path.basename(i), pinjie)
