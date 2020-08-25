#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

typedef struct Pixel 
{
    int x, y;
    int data;
}Pixel;

bool structCmp(const Pixel &a, const Pixel &b) 
{
    return a.data > b.data;//descending降序
}

Mat minFilter(Mat srcImage, int kernelSize);
void makeDepth32f(Mat& source, Mat& output);
void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon);
Mat getTransmission_dark(Mat& srcimg, Mat& darkimg, int *array, int windowsize);
Mat recover(Mat& srcimg, Mat& t, float *array, int windowsize);

int main() 
{
    string loc = "/home/cheng/Documents/practice_py/cv/dehaze/dehaze-master/image/city.png";
    double scale = 1.0;
    string name = "forest";
    clock_t start, finish;
    double duration;

    cout << "A defog program" << endl
        << "----------------" << endl;

    Mat image = imread(loc);
    Mat resizedImage;
    int originRows = image.rows;
    int originCols = image.cols;

    if (scale < 1.0) 
    {
        resize(image, resizedImage, Size(originCols * scale, originRows * scale));
    }
    else 
    {

        scale = 1.0;
        resizedImage = image;
    }

    int rows = resizedImage.rows;
    int cols = resizedImage.cols;
    Mat convertImage;
    resizedImage.convertTo(convertImage, CV_32FC3, 1 / 255.0, 0);
    int kernelSize = 15 ? max((rows * 0.01), (cols * 0.01)) : 15 < max((rows * 0.01), (cols * 0.01));
    //int kernelSize = 15;
    int parse = kernelSize / 2;
    Mat darkChannel(rows, cols, CV_8UC1);
    Mat normalDark(rows, cols, CV_32FC1);
    int nr = rows;
    int nl = cols;
    float b, g, r;
    start = clock();
    cout << "generating dark channel image." << endl;
    if (resizedImage.isContinuous()) 
    {
        nl = nr * nl;
        nr = 1;
    }
    for (int i = 0; i < nr; i++) 
    {
        float min;
        const uchar* inData = resizedImage.ptr<uchar>(i);
        uchar* outData = darkChannel.ptr<uchar>(i);
        for (int j = 0; j < nl; j++) 
        {
            b = *inData++;
            g = *inData++;
            r = *inData++;
            min = b > g ? g : b;
            min = min > r ? r : min;
            *outData++ = min;
        }
    }
    darkChannel = minFilter(darkChannel, kernelSize);

    imshow("darkChannel", darkChannel);
    cout << "dark channel generated." << endl;

    //estimate Airlight
    //开一个结构体数组存暗通道，再sort，取最大0.1%，利用结构体内存储的原始坐标在原图中取点
    cout << "estimating airlight." << endl;
    rows = darkChannel.rows, cols = darkChannel.cols;
    int pixelTot = rows * cols * 0.001;
    int *A = new int[3];
    Pixel *toppixels, *allpixels;
    toppixels = new Pixel[pixelTot];
    allpixels = new Pixel[rows * cols];


    for (unsigned int r = 0; r < rows; r++) 
    {
        const uchar *data = darkChannel.ptr<uchar>(r);
        for (unsigned int c = 0; c < cols; c++) 
        {
            allpixels[r*cols + c].data = *data;
            allpixels[r*cols + c].x = r;
            allpixels[r*cols + c].y = c;
        }
    }
    std::sort(allpixels, allpixels + rows * cols, structCmp);

    memcpy(toppixels, allpixels, pixelTot * sizeof(Pixel));

    float A_r, A_g, A_b, avg, maximum = 0;
    int idx, idy, max_x, max_y;
    for (int i = 0; i < pixelTot; i++) 
    {
        idx = allpixels[i].x; idy = allpixels[i].y;
        const uchar *data = resizedImage.ptr<uchar>(idx);
        data += 3 * idy;
        A_b = *data++;
        A_g = *data++;
        A_r = *data++;
        //cout << A_r << " " << A_g << " " << A_b << endl;
        avg = (A_r + A_g + A_b) / 3.0;
        if (maximum < avg) 
        {
            maximum = avg;
            max_x = idx;
            max_y = idy;
        }
    }

    delete[] toppixels;
    delete[] allpixels;

    for (int i = 0; i < 3; i++) 
    {
        A[i] = resizedImage.at<Vec3b>(max_x, max_y)[i];
    }
    cout << "airlight estimated as: " << A[0] << ", " << A[1] << ", " << A[2] << endl;
    //cout << endl;

    //暗通道归一化操作（除A）
    //(I / A)
    cout << "start normalization of input image I." << endl;
    float tmp_A[3];
    tmp_A[0] = A[0] / 255.0;
    tmp_A[1] = A[1] / 255.0;
    tmp_A[2] = A[2] / 255.0;
    for (int i = 0; i < nr; i++) 
    {
        float min = 1.0;
        const float* inData = convertImage.ptr<float>(i);
        float* outData = normalDark.ptr<float>(i);
        for (int j = 0; j < nl; j++) 
        {
            b = *inData++ / tmp_A[0];
            g = *inData++ / tmp_A[1];
            r = *inData++ / tmp_A[2];
            min = b > g ? g : b;
            min = min > r ? r : min;
            *outData++ = min;
        }
    }
    cout << "normalization finished." << endl << "generating relative dark channel image." << endl;
    //暗通道最小滤波
    normalDark = minFilter(normalDark, kernelSize);
    cout << "dark channel image generated." << "start estimating transmission and guided image filtering." << endl;
    imshow("normal", normalDark);
    int kernelSizeTrans = std::max(3, kernelSize);
    //求t与将t进行导向滤波

    Mat trans = getTransmission_dark(convertImage, normalDark, A, kernelSizeTrans);
    cout << "tansmission estimated and guided filtered." << endl;
    imshow("filtered t", trans);
    cout << "start recovering." << endl;
    Mat finalImage = recover(convertImage, trans, tmp_A, kernelSize);
    cout << "recovering finished." << endl;
    Mat resizedFinal;
    if (scale < 1.0) 
    {
        resize(finalImage, resizedFinal, Size(originCols, originRows));
        imshow("final", resizedFinal);
    }
    else 
    {
        imshow("final", finalImage);
    }
    finish = clock();
    duration = (double)(finish - start);
    cout << "defog used " << duration << "ms time;" << endl;
    waitKey(0);

    finalImage.convertTo(finalImage, CV_8UC3, 255);
    imwrite(name + "_refined.png", finalImage);
    destroyAllWindows();
    image.release();
    resizedImage.release();
    convertImage.release();
    darkChannel.release();
    trans.release();
    finalImage.release();
    return 0;
}

Mat minFilter(Mat srcImage, int kernelSize) 
{
    int radius = kernelSize / 2;

    int srcType = srcImage.type();
    int targetType = 0;
    if (srcType % 8 == 0) 
    {
        targetType = 0;
    }
    else 
    {
        targetType = 5;
    }
    Mat ret(srcImage.rows, srcImage.cols, targetType);
    Mat parseImage;
    copyMakeBorder(srcImage, parseImage, radius, radius, radius, radius, BORDER_REPLICATE);
    for (unsigned int r = 0; r < srcImage.rows; r++) 
    {
        float *fOutData = ret.ptr<float>(r);
        uchar *uOutData = ret.ptr<uchar>(r);
        for (unsigned int c = 0; c < srcImage.cols; c++) 
        {
            Rect ROI(c, r, kernelSize, kernelSize);
            Mat imageROI = parseImage(ROI);
            double minValue = 0, maxValue = 0;
            Point minPt, maxPt;
            minMaxLoc(imageROI, &minValue, &maxValue, &minPt, &maxPt);
            if (!targetType) 
            {
                *uOutData++ = (uchar)minValue;
                continue;
            }
            *fOutData++ = minValue;
        }
    }
    return ret;
}

void makeDepth32f(Mat& source, Mat& output)
{
    if ((source.depth() != CV_32F) > FLT_EPSILON)
        source.convertTo(output, CV_32F);
    else
        output = source;
}

void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
{
    CV_Assert(radius >= 2 && epsilon > 0);
    CV_Assert(source.data != NULL && source.channels() == 1);
    CV_Assert(guided_image.channels() == 1);
    CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

    Mat guided;
    if (guided_image.data == source.data)
    {
        //make a copy
        guided_image.copyTo(guided);
    }
    else
    {
        guided = guided_image;
    }

    //将输入扩展为32位浮点型，以便以后做乘法
    Mat source_32f, guided_32f;
    makeDepth32f(source, source_32f);
    makeDepth32f(guided, guided_32f);

    //计算I*p和I*I
    Mat mat_Ip, mat_I2;
    multiply(guided_32f, source_32f, mat_Ip);
    multiply(guided_32f, guided_32f, mat_I2);

    //计算各种均值
    Mat mean_p, mean_I, mean_Ip, mean_I2;
    Size win_size(2 * radius + 1, 2 * radius + 1);
    boxFilter(source_32f, mean_p, CV_32F, win_size);
    boxFilter(guided_32f, mean_I, CV_32F, win_size);
    boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
    boxFilter(mat_I2, mean_I2, CV_32F, win_size);

    //计算Ip的协方差和I的方差
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_I2 - mean_I.mul(mean_I);
    var_I += epsilon;

    //求a和b
    Mat a, b;
    divide(cov_Ip, var_I, a);
    b = mean_p - a.mul(mean_I);

    //对包含像素i的所有a、b做平均
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_32F, win_size);
    boxFilter(b, mean_b, CV_32F, win_size);

    //计算输出 (depth == CV_32F)
    output = mean_a.mul(guided_32f) + mean_b;
}

Mat getTransmission_dark(Mat& srcimg, Mat& darkimg, int *array, int windowsize)
{
    //t = 1 - omega * minfilt(I / A);
    float avg_A;
    //convertImage是一个CV_32FC3的图
    if (srcimg.type() % 8 == 0) {
        avg_A = (array[0] + array[1] + array[2]) / 3.0;
    }
    else {
        avg_A = (array[0] + array[1] + array[2]) / (3.0 * 255.0);
    }
    float w = 0.95;
    int radius = windowsize / 2;
    int nr = srcimg.rows, nl = srcimg.cols;
    Mat transmission(nr, nl, CV_32FC1);

    for (int k = 0; k<nr; k++) 
    {
        const float* inData = darkimg.ptr<float>(k);
        float* outData = transmission.ptr<float>(k);
        float pix[3] = { 0 };
        for (int l = 0; l<nl; l++)
        {
            *outData++ = 1.0 - w * *inData++;
        }
    }
    imshow("t", transmission);

    Mat trans(nr, nl, CV_32FC1);
    Mat graymat(nr, nl, CV_8UC1);
    Mat graymat_32F(nr, nl, CV_32FC1);

    if (srcimg.type() % 8 != 0) 
    {
        cvtColor(srcimg, graymat_32F, COLOR_BGR2GRAY);
        guidedFilter(transmission, graymat_32F, trans, 6 * windowsize, 0.001);
    }
    else 
    {
        cvtColor(srcimg, graymat, COLOR_BGR2GRAY);

        for (int i = 0; i < nr; i++) 
        {
            const uchar* inData = graymat.ptr<uchar>(i);
            float* outData = graymat_32F.ptr<float>(i);
            for (int j = 0; j < nl; j++)
                *outData++ = *inData++ / 255.0;
        }
        guidedFilter(transmission, graymat_32F, trans, 6 * windowsize, 0.001);
    }
    return trans;
}

Mat recover(Mat& srcimg, Mat& t, float *array, int windowsize)
{
    //J(x) = (I(x) - A) / max(t(x), t0) + A;
    int radius = windowsize / 2;
    int nr = srcimg.rows, nl = srcimg.cols;
    float tnow = t.at<float>(0, 0);
    float t0 = 0.1;
    Mat finalimg = Mat::zeros(nr, nl, CV_32FC3);
    float val = 0;

    //Be aware that transmission is a grey image
    //srcImg is a color image
    //finalImg is a color image
    //Mat store color image a pixel per 3 position
    //store grey image a pixel per 1 position
    for (unsigned int r = 0; r < nr; r++) {
        const float* transPtr = t.ptr<float>(r);
        const float* srcPtr = srcimg.ptr<float>(r);
        float* outPtr = finalimg.ptr<float>(r);
        for (unsigned int c = 0; c < nl; c++) {
            //transmission image is grey, so only need 
            //to move once per calculation, using index 
            //c(a.k.a. columns) to move is enough 
            tnow = *transPtr++;
            tnow = std::max(tnow, t0);
            for (int i = 0; i < 3; i++) {
                //so to calculate every color channel per pixel
                //move the ptr once after one calculation.
                //after 3 times move, calculation for a pixel is done
                val = (*srcPtr++ - array[i]) / tnow + array[i];
                *outPtr++ = val + 10 / 255.0;
            }
        }
    }
    cout << finalimg.size() << endl;
    return finalimg;
}