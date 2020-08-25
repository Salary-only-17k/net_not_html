#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat GF_smooth(Mat& src, int s, double epsilon);
    Mat staticMin(Mat& I, int s, double eeps, double alpha);
    int est_air(Mat& src, int s, double *A_r, double *A_g, double *A_b);
    Mat est_trans_fast(Mat& src, int s, double eeps, double k, double A_r, double A_g, double A_b);
    Mat rmv_haze(Mat& src, Mat& t, double A_r, double A_g, double A_b);

    string loc = "/home/cheng/Documents/practice_py/cv/dehaze/dehaze-master/image/city.png";
    string name = "forest";
    clock_t start, finish;
    double duration;

    cout << "A defog program" << endl
         << "---------------" << endl;

    start = clock();
        Mat image = imread(loc);
    imshow("hazyimage", image);
    cout << "input hazy image" << endl;
    
    int s = 15;
    double eeps = 0.002, omega = 0.95;
    double air_r, air_g, air_b;
    est_air(image, s, &air_r, &air_g, &air_b);
    cout << "airlight estimation as:" << air_r << ", " << air_g << ", " << air_b << endl;

    Mat t;
    t=est_trans_fast(image, s, eeps, omega, air_r, air_g, air_b);

    Mat dehazedimage;
    dehazedimage = rmv_haze(image, t, air_r, air_g, air_b);
    imshow("dehaze", dehazedimage);
    finish = clock();
    duration = (double)(finish - start);
    cout << "defog used" << duration << "ms time;" << endl;
    waitKey(0);

    imwrite(name + "_refined.png", dehazedimage);
    destroyAllWindows();
    image.release();
    dehazedimage.release();
    return 0;
}

//---------------------- GUIDED FILTER -------------------//
Mat GF_smooth(Mat& src, int s, double epsilon)
{
    src.convertTo(src, CV_64FC1);

    Mat mean_I;
    blur(src, mean_I, Size(s, s), Point(-1, -1));

    Mat II = src.mul(src);
    Mat var_I;
    blur(II, var_I, Size(s, s), Point(-1, -1));
    var_I = var_I - mean_I.mul(mean_I);

    Mat a = var_I / ((var_I + epsilon));
    Mat b = mean_I - a.mul(mean_I);

    Mat mean_a;
    blur(a, mean_a, Size(s, s), Point(-1, -1));
    Mat mean_b;
    blur(b, mean_b, Size(s, s), Point(-1, -1));

    return mean_a.mul(src) + mean_b;
}


Mat staticMin(Mat& I, int s, double eps, double alpha)
{
    Mat mean_I = GF_smooth(I, s, eps);

    Mat var_I;
    blur((I - mean_I).mul(I - mean_I), var_I, Size(s, s), Point(-1, -1));

    Mat mean_var_I;
    blur(var_I, mean_var_I, Size(s, s), Point(-1, -1));

    Mat z_I;
    sqrt(mean_var_I, z_I);

    return mean_I - alpha*z_I;
}


//---------------------- DEHAZING FUNCTIONS -------------------//
int est_air(Mat& src, int s, double *A_r, double *A_g, double *A_b)
    {
        /// Split RGB to channels
        src.convertTo(src, CV_64FC3);
        vector<Mat> channels(3);
        split(src, channels);        // separate color channels

        Mat R = channels[2];
        Mat G = channels[1];
        Mat B = channels[0];

        Mat Im = min(min(R, G), B);

        /// Estimate airlight
        Mat blur_Im;
        blur(Im, blur_Im, Size(s, s), Point(-1, -1));

        int maxIdx[2] = { 0, 0 };
        minMaxIdx(blur_Im, NULL, NULL, NULL, maxIdx);

        int width = R.cols;
        *A_r = ((double*)R.data)[maxIdx[0] * R.cols + maxIdx[1]];
        *A_g = ((double*)G.data)[maxIdx[0] * R.cols + maxIdx[1]];
        *A_b = ((double*)B.data)[maxIdx[0] * R.cols + maxIdx[1]];

        return 0;
    }

    Mat est_trans_fast(Mat& src, int s, double eeps, double k, double A_r, double A_g, double A_b)
    {
        /// Split RGB to channels
        src.convertTo(src, CV_64FC3);
        vector<Mat> channels(3);
        split(src, channels);        // separate color channels

        Mat R = channels[2];
        Mat G = channels[1];
        Mat B = channels[0];

        /// Estimate transmission
        Mat R_n = R / A_r;
        Mat G_n = G / A_g;
        Mat B_n = B / A_b;

        Mat Im = min(min(R_n, G_n), B_n);

        eeps = (3 * 255 / (A_r + A_g + A_b))*(3 * 255 / (A_r + A_g + A_b))*eeps;
        double alpha = 2;
        Mat z_Im = staticMin(Im, s, eeps, alpha);

        return min(max(0.001, 1 - k*z_Im), 1);
    }


    Mat rmv_haze(Mat& src, Mat& t, double A_r, double A_g, double A_b)
    {
        /// Split RGB to channels
        src.convertTo(src, CV_64FC3);
        vector<Mat> channels(3);
        split(src, channels);        // separate color channels

        Mat R = channels[2];
        Mat G = channels[1];
        Mat B = channels[0];

        /// Remove haze
        channels[2] = (R - A_r) / t + A_r;
        channels[1] = (G - A_g) / t + A_g;
        channels[0] = (B - A_b) / t + A_b;

        Mat dst;
        merge(channels, dst);

        dst.convertTo(dst, CV_8UC3);
        return dst;
    }