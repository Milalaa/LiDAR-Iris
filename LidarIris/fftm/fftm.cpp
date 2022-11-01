#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//----------------------------------------------------------
// Recombinate image quaters
//----------------------------------------------------------
void Recomb(Mat &src, Mat &dst)
{
    int cx = src.cols >> 1;
    int cy = src.rows >> 1;
    Mat tmp;
    tmp.create(src.size(), src.type());
    src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
    src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));
    src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
    src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
    dst = tmp;
}
//----------------------------------------------------------
// 2D Forward FFT
//----------------------------------------------------------
void ForwardFFT(Mat &Src, Mat *FImg, )
{
    //    将输入图像扩展到最佳尺寸，边界用0填充
    //    离散傅里叶变换的运行速度与图像的大小有很大的关系，当图像的尺寸使2，3，5的整数倍时，计算速度最快
    //    为了达到快速计算的目的，经常通过添加新的边缘像素的方法获取最佳图像尺寸
    //    函数getOptimalDFTSize()用于返回最佳尺寸，copyMakeBorder()用于填充边缘像素
    int M = getOptimalDFTSize(Src.rows);    // 返回行的最佳尺寸
    int N = getOptimalDFTSize(Src.cols);    // 返回列的最佳尺寸
    Mat padded;
    /* void copyMakeBorder(InputArray  src,        //输入图像
		    OutputArray dst,        //输出图像
		    int top,                //上边界添加的像素行数
		    int bottom,             //下边界添加的像素行数
		    int left,               //左边界添加的像素列数
		    int right,              //右边界添加的像素列数
		    int borderType,         //表示边界的类型
		    const Scalar& value=Scalar()//表示如果边界的类型是BORDER_CONSTANT时边界的颜色值 )
            BORDER_CONSTANT:常量复制：          iiiiii|abcdefgh|iiiiiii(i的值由最后一个参数 const Scalar& value=Scalar()确定，如Scalar::all(0) )*/
    copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));
    
    //    为傅立叶变换的结果分配存储空间
    //    将plannes数组组合成一个多通道的数组，两个同搭配，分别保存实部和虚部
    //    傅里叶变换的结果是复数，这就是说对于每个图像原像素值，会有两个图像值
    //    此外，频域值范围远远超过图象值范围，因此至少将频域储存在float中
    //    所以我们将输入图像转换成浮点型，并且多加一个额外通道来存储复数部分
    //   CV_32F - 32位浮点数
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexImg;
    //  merge将多个阵列合并为一个多通道阵列。也就是说，输出数组的每个元素都是输入数组元素的串联，其中第i个输入数组的元素被视为mv[i]。
    /* CV_EXPORTS void merge(const Mat* mv, size_t count, OutputArray dst);
         mv 要合并的矩阵输入数组；mv中的所有矩阵必须具有相同的大小和深度。
         count 当mv是纯C数组时，计数输入矩阵的数目；它必须大于零。
         dst 输出数组，大小和深度与mv[0]相同；通道数将等于参数计数。*/
    merge(planes, 2, complexImg);
    
    //    进行离散傅立叶变换
    //    CV_EXPORTS_W void dft(InputArray src, OutputArray dst, int flags = 0, int nonzeroRows = 0);
    dft(complexImg, complexImg);
    
    //    将复数转化为幅值
    /*     CV_EXPORTS void split(const Mat& src, Mat* mvbegin);
            src 输入多通道阵列
            mvbegin 输出数组；数组的数目必须与src匹配。如果需要，阵列本身会被重新分配。
    */
    split(complexImg, planes);
    //剪切和重分布幅度图像限
	//如果有奇数行或奇数列，进行频谱裁剪
    planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));    // Re
    planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));    // Im
    if (do_recomb)
    {
        Recomb(planes[0], planes[0]);
        Recomb(planes[1], planes[1]);
    }
    planes[0] /= float(M*N);
    planes[1] /= float(M*N);
    FImg[0] = planes[0].clone();
    FImg[1] = planes[1].clone();
}
//----------------------------------------------------------
// 2D inverse FFT
//----------------------------------------------------------
void InverseFFT(Mat *FImg, Mat &Dst, bool do_recomb = true)
{
    if (do_recomb)
    {
        Recomb(FImg[0], FImg[0]);
        Recomb(FImg[1], FImg[1]);
    }
    Mat complexImg;
    merge(FImg, 2, complexImg);
    idft(complexImg, complexImg);
    split(complexImg, FImg);
    Dst = FImg[0].clone();
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
void highpass(Size sz, Mat& dst)
{
    Mat a = Mat(sz.height, 1, CV_32FC1);
    Mat b = Mat(1, sz.width, CV_32FC1);

    float step_y = CV_PI / sz.height;
    float val = -CV_PI*0.5;

    for (int i = 0; i < sz.height; ++i)
    {
        a.at<float>(i) = cos(val);
        val += step_y;
    }

    val = -CV_PI*0.5;
    float step_x = CV_PI / sz.width;
    for (int i = 0; i < sz.width; ++i)
    {
        b.at<float>(i) = cos(val);
        val += step_x;
    }

    Mat tmp = a*b;
    dst = (1.0 - tmp).mul(2.0 - tmp);
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
float logpolar(Mat& src, Mat& dst)
{
    float radii = src.cols;
    float angles = src.rows;
    Point2f center(src.cols / 2, src.rows / 2);
    float d = norm(Vec2f(src.cols - center.x, src.rows - center.y));
    float log_base = pow(10.0, log10(d) / radii);
    float d_theta = CV_PI / (float)angles;
    float theta = CV_PI / 2.0;
    float radius = 0;
    Mat map_x(src.size(), CV_32FC1);
    Mat map_y(src.size(), CV_32FC1);
    for (int i = 0; i < angles; ++i)
    {
        for (int j = 0; j < radii; ++j)
        {
            radius = pow(log_base, float(j));
            float x = radius * sin(theta) + center.x;
            float y = radius * cos(theta) + center.y;
            map_x.at<float>(i, j) = x;
            map_y.at<float>(i, j) = y;
        }
        theta += d_theta;
    }
    remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    return log_base;
}
//-----------------------------------------------------------------------------------------------------
// As input we need equal sized images, with the same aspect ratio,
// scale difference should not exceed 1.8 times.
//-----------------------------------------------------------------------------------------------------
RotatedRect LogPolarFFTTemplateMatch(Mat& im0, Mat& im1/*, double canny_threshold1, double canny_threshold2*/)
{
    // Accept 1 or 3 channel CV_8U, CV_32F or CV_64F images.
    CV_Assert((im0.type() == CV_8UC1) || (im0.type() == CV_8UC3) ||
        (im0.type() == CV_32FC1) || (im0.type() == CV_32FC3) ||
        (im0.type() == CV_64FC1) || (im0.type() == CV_64FC3));

    CV_Assert(im0.rows == im1.rows && im0.cols == im1.cols);

    CV_Assert(im0.channels() == 1 || im0.channels() == 3 || im0.channels() == 4);

    CV_Assert(im1.channels() == 1 || im1.channels() == 3 || im1.channels() == 4);

    //Mat im0_tmp = im0.clone();
    //Mat im1_tmp = im1.clone();
    if (im0.channels() == 3)
    {
        cvtColor(im0, im0, cv::COLOR_BGR2GRAY);
    }

    if (im0.channels() == 4)
    {
        cvtColor(im0, im0, cv::COLOR_BGRA2GRAY);
    }

    if (im1.channels() == 3)
    {
        cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
    }

    if (im1.channels() == 4)
    {
        cvtColor(im1, im1, cv::COLOR_BGRA2GRAY);
    }

    if (im0.type() == CV_32FC1)
    {
       im0.convertTo(im0, CV_8UC1, 255.0);
    }

    if (im1.type() == CV_32FC1)
    {
       im1.convertTo(im1, CV_8UC1, 255.0);
    }

    if (im0.type() == CV_64FC1)
    {
        im0.convertTo(im0, CV_8UC1, 255.0);
    }

    if (im1.type() == CV_64FC1)
    {
        im1.convertTo(im1, CV_8UC1, 255.0);
    }


    // Canny(im0, im0, canny_threshold1, canny_threshold2); // you can change this
    // Canny(im1, im1, canny_threshold1, canny_threshold2);
    
    // Ensure both images are of CV_32FC1 type
    im0.convertTo(im0, CV_32FC1, 1.0 / 255.0);
    im1.convertTo(im1, CV_32FC1, 1.0 / 255.0);

    Mat F0[2], F1[2];
    Mat f0, f1;
    ForwardFFT(im0, F0);
    ForwardFFT(im1, F1);
    magnitude(F0[0], F0[1], f0);
    magnitude(F1[0], F1[1], f1);

    // Create filter 
    Mat h;
    highpass(f0.size(), h);

    // Apply it in freq domain
    f0 = f0.mul(h);
    f1 = f1.mul(h);

    float log_base;
    Mat f0lp, f1lp;

    log_base = logpolar(f0, f0lp);
    log_base = logpolar(f1, f1lp);

    // Find rotation and scale
    Point2d rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);

    float angle = 180.0 * rotation_and_scale.y / f0lp.rows;
    float scale = pow(log_base, rotation_and_scale.x);
    // --------------
    if (scale > 1.8)
    {
        rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);
        angle = -180.0 * rotation_and_scale.y / f0lp.rows;
        scale = 1.0 / pow(log_base, rotation_and_scale.x);
        if (scale > 1.8)
        {
            cout << "Images are not compatible. Scale change > 1.8" << endl;
            return RotatedRect();
        }
    }
    // --------------
    if (angle < -90.0)
    {
        angle += 180.0;
    }
    else if (angle > 90.0)
    {
        angle -= 180.0;
    }

    // Now rotate and scale fragment back, then find translation
    Mat rot_mat = getRotationMatrix2D(Point(im1.cols / 2, im1.rows / 2), angle, 1.0 / scale);

    // rotate and scale
    Mat im1_rs;
    warpAffine(im1, im1_rs, rot_mat, im1.size());

    // find translation
    Point2d tr = cv::phaseCorrelate(im1_rs, im0);

    // compute rotated rectangle parameters
    RotatedRect rr;
    rr.center = tr + Point2d(im0.cols / 2, im0.rows / 2);
    rr.angle = -angle;
    rr.size.width = im1.cols / scale;
    rr.size.height = im1.rows / scale;

    //im0 = im0_tmp.clone();
    //im1 = im1_tmp.clone();

    return rr;
}

RotatedRect FFTMatch(const Mat& im0, const Mat& im1)
{
    Mat im0_tmp = im0.clone();
    Mat im1_tmp = im1.clone();
    return LogPolarFFTTemplateMatch(im0_tmp, im1_tmp);
}
