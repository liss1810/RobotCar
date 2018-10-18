#include "CornerFinder.h"
#include "Common_GQ.h"
#include <numeric>
#include <algorithm>
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_EPS
#define M_EPS 2.2204e-16
#endif

using namespace cv;
using namespace std;
using namespace Common_GQ;

int GlobalScale(cv::Mat& I_64FC1);

cv::Mat Conv2(const cv::Mat &img, const cv::Mat& ikernel);

int AdaptImageAdjust(const cv::Mat& _im, cv::Mat& imadj_, const int* _win = NULL, const double* _th = NULL);

int GetEdges(const cv::Mat& _im, cv::Mat& imgedge_, cv::Mat& ix_, cv::Mat& iy_);

int HarrisCorners(const cv::Mat& _imadj, cv::Mat& imgh_, const int* _hwin = NULL);

int AdaptStatus(const cv::Mat& _imgh, cv::Mat& mimg_, cv::Mat& stdv_);

int AdaptStatus(const cv::Mat& _imgh, const int& win, cv::Mat& mimg_, cv::Mat& stdv_);

std::vector<cv::Point2d> GetControlPoints(const cv::Mat& _imgh, const cv::Mat& _mimg, const cv::Mat& _stdv, const double& _th);

int SweepMatrix(const cv::Mat& img, cv::Mat& sweepmatx, cv::Mat& sweepmaty);

std::vector<int> FindNearestPoints(const cv::Point2d& point, const std::vector<cv::Point2d>& crnrs, const int num = 1, bool same = false);

cv::Point2d FindNearestPoints(const int index, const std::vector<cv::Point2d>& crnrs);

int GetWin(const cv::Mat& img, const int index, const std::vector<cv::Point2d>& crnrs);

int AdjustImage(cv::Mat& img, const double* imeadjsca = NULL);

int CircleSweep(const cv::Mat& imgedge, const cv::Mat& sweepmatx, const cv::Mat& sweepmaty,
	cv::Mat& theta, cv::Mat& edgevalue, cv::Mat& thetasmd, cv::Mat& edgevaluesmd);

int Cadj(const cv::Mat& edgevalue, std::vector<std::pair<int, double>>& v, double& v_max, double& v_min);

int PeakDet(const cv::Mat& edgevalue, std::vector<std::pair<int, double>>& maxtab);

double JudgeValidCorner(cv::Mat& imgcrop, cv::Mat& imgedgecrop, const cv::Mat& sweepmatxcrop, const cv::Mat& sweepmatycrop, std::vector<int>& plocs);

int ChessCornerFilter(const C_Pyramid& _pyr, const std::vector<cv::Point2d>& ctcrnrpts, std::vector<cv::Point2d>& crnrs, int &nocrnrs, std::vector<std::vector<int>>& peaklocs);

int LandMarkCornerFilter(const C_Pyramid& _pyr, const std::vector<cv::Point2d>& ctcrnrpts, std::vector<cv::Point2d>& crnrs, int &nocrnrs, std::vector<std::vector<int>>& peaklocs);

double Deg2Rad(const double angdegs);

bool AngleProx(const double ang1, const double ang2, const double th);

int GetGrid(const std::vector<cv::Point2d>& crnrs, const std::vector<cv::Point2d>& crnrpts, const std::vector<std::vector<int>>& peaklocs, const C_Pyramid& _pyr, cv::Mat& crnrsgrid);

int FilterGrid(const cv::Mat& crnrsgrid, cv::Mat& crnrsgridfil);

int AdjustGridDirection(cv::Mat& crnrsgridfil);

bool Mat2Points(const cv::Mat &matx, const cv::Mat &maty, std::vector<cv::Point2d> &points, bool is_row);

int FixMissCorners(const cv::Mat& crnrsgridfil, const cv::Size img_size, cv::Mat& gridfullrect, int& nointerpolations);

int FitLine2D(const std::vector<cv::Point2d> &points, double &k, double &d);

//************************************
// Method:    AdjustGridOrigin
// FullName:  AdjustGridOrigin
// Access:    public 
// Returns:   int
// Qualifier: 调整棋盘格方向，保证起点在左上角
// Parameter: cv::Mat & gridfullrect
// Parameter: int origin_pos 起点位置，0为左上，1为右上，2为左下，3为右下
//************************************
int AdjustGridOrigin(cv::Mat& gridfullrect, int origin_pos=0);

void InvPointsXY(std::vector<cv::Point2d> &out);

int GlobalScale(cv::Mat& I_64FC1)
{
	double p_min, p_max;
	minMaxLoc(I_64FC1, &p_min, &p_max);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < I_64FC1.rows; r++)
	{
		for (int c = 0; c < I_64FC1.cols; c++)
		{
			double& p = I_64FC1.at<double>(r, c);
			p = (p - p_min) / (p_max - p_min);
		}
	}
	return 0;
}

int AdaptImageAdjust(const cv::Mat& _img, cv::Mat& imadj_, const int* _win, const double* _th)
{
	int win, th;
	if (NULL == _win)
		win = (int)round(min((double)_img.rows / 5.0, (double)_img.cols / 5.0));

	if (NULL == _th)
		th = 1;

	Mat img = _img;
	Mat ming, stdv;
	AdaptStatus(img, win, ming, stdv);

	Mat imax, imin;
	addWeighted(ming, 1, stdv, th, 0, imax);
	addWeighted(ming, 1, stdv, -th, 0, imin);

	// clip
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < imax.rows; r++)
	{
		for (int c = 0; c < imax.cols; c++)
		{
			double& v_max = imax.at<double>(r, c);
			double& v_min = imin.at<double>(r, c);
			v_max = v_max > 1.0 ? 1.0 : v_max;
			v_min = v_min < 0.0 ? 0.0 : v_min;
		}
	}

	imadj_ = Mat(img.rows, img.cols, CV_64FC1);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++)
		{
			double& p = img.at<double>(r, c);
			double& p_out = imadj_.at<double>(r, c);
			double& v_max = imax.at<double>(r, c);
			double& v_min = imin.at<double>(r, c);
			p_out = (p - v_min) / (v_max - v_min);
		}
	}

	// adjust for clipping and saturation
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < imadj_.rows; r++)
	{
		for (int c = 0; c < imadj_.cols; c++)
		{
			double& p = imadj_.at<double>(r, c);
			p = p > 1.0 ? 1.0 : p;
			p = p < 0.0 ? 0.0 : p;
		}
	}

	return 0;
}

cv::Mat Conv2(const cv::Mat &img, const cv::Mat& ikernel)
{
	Mat dest;
	Mat kernel;
	flip(ikernel, kernel, -1);
	Mat source = img;
	Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = BORDER_CONSTANT;
	filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);

	return dest;
}

int GetEdges(const cv::Mat& _im, cv::Mat& imgedge_, cv::Mat& ix_, cv::Mat& iy_)
{
	Mat dx = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat dy = dx.t();

	ix_ = Conv2(_im, dx);
	iy_ = Conv2(_im, dy);

	imgedge_ = Mat(_im.rows, _im.cols, CV_64FC1);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < imgedge_.rows; r++)
	{
		for (int c = 0; c < imgedge_.cols; c++)
		{
			double& p_x = ix_.at<double>(r, c);
			double& p_y = iy_.at<double>(r, c);
			double& p_e = imgedge_.at<double>(r, c);
			p_e = pow(p_x * p_x + p_y * p_y, 0.5);
		}
	}

	GlobalScale(imgedge_);

	return 0;
}

int HarrisCorners(const cv::Mat& _imadj, cv::Mat& imgh, const int* hwin)
{
	int win;
	if (NULL == hwin)
		win = (int)round((double)min(_imadj.rows, _imadj.cols) / 140);
	else
		win = *hwin;

	Mat dx = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat dy = dx.t();

	Mat ix, iy;
	ix = Conv2(_imadj, dx);
	iy = Conv2(_imadj, dy);

	Mat m = Mat::ones(Size(win, win), CV_64FC1);
	m = m / (win * win);
	Mat a, b, c;

	Mat ixix, iyiy, ixiy;
	multiply(ix, ix, ixix);
	multiply(iy, iy, iyiy);
	multiply(ix, iy, ixiy);
	a = Conv2(ixix, m);
	b = Conv2(iyiy, m);
	c = Conv2(ixiy, m);


	imgh = Mat(_imadj.rows, _imadj.cols, CV_64FC1);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int row = 0; row < imgh.rows; row++)
	{
		for (int col = 0; col < imgh.cols; col++)
		{
			double& p_A = a.at<double>(row, col);
			double& p_B = b.at<double>(row, col);
			double& p_C = c.at<double>(row, col);
			double& p_O = imgh.at<double>(row, col);
			p_O = (p_A * p_B - p_C * p_C) / (p_A + p_B + M_EPS);
		}
	}
	GlobalScale(imgh);

	return 0;
}

int AdaptStatus(const cv::Mat& _imgh, cv::Mat& mimg_, cv::Mat& stdv_)
{
	int win = (int)round((double)min(_imgh.rows, _imgh.cols) / 5);

	AdaptStatus(_imgh, win, mimg_, stdv_);
	return 0;
}

int AdaptStatus(const cv::Mat& _imgh, const int& win, cv::Mat& mimg_, cv::Mat& stdv_)
{
	mimg_ = Mat(_imgh.rows, _imgh.cols, CV_64FC1);
	stdv_ = Mat(_imgh.rows, _imgh.cols, CV_64FC1);

	// create integral image, integral image allows for quicker execution time for large window sizes.
	Mat intimg, intimg2;
	integral(_imgh, intimg, intimg2);

	int hwin = (int)floor(win / 2);

	// pad integral images

	copyMakeBorder(intimg, intimg, hwin, 0, hwin, 0, BORDER_CONSTANT, Scalar(0));
	copyMakeBorder(intimg, intimg, 0, hwin, 0, hwin, BORDER_REPLICATE);
	copyMakeBorder(intimg2, intimg2, hwin, 0, hwin, 0, BORDER_CONSTANT, Scalar(0));
	copyMakeBorder(intimg2, intimg2, 0, hwin, 0, hwin, BORDER_REPLICATE);

	Mat simg = intimg(Range(0, _imgh.rows), Range(0, _imgh.cols))
		+ intimg(Range(win, _imgh.rows + win), Range(win, _imgh.cols + win))
		- intimg(Range(0, _imgh.rows), Range(win, _imgh.cols + win))
		- intimg(Range(win, _imgh.rows + win), Range(0, _imgh.cols));
	Mat simg2 = intimg2(Range(0, _imgh.rows), Range(0, _imgh.cols))
		+ intimg2(Range(win, _imgh.rows + win), Range(win, _imgh.cols + win))
		- intimg2(Range(0, _imgh.rows), Range(win, _imgh.cols + win))
		- intimg2(Range(win, _imgh.rows + win), Range(0, _imgh.cols));

	int n = win * win;
	mimg_ = simg / n;
	Mat ming2;
	multiply(mimg_, mimg_, ming2);
	Mat vari = (simg2 - ming2 * n) / (n - 1);

	double varith = 0.02 * 0.02;
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < vari.rows; r++)
	{
		for (int c = 0; c < vari.cols; c++)
		{
			double& p = vari.at<double>(r, c);

			p = p < varith ? varith : p;
		}
	}
	pow(vari, 0.5, stdv_);

	return 0;
}

int SweepMatrix(const cv::Mat& img, cv::Mat& sweepmatx, cv::Mat& sweepmaty)
{
	int win = min(img.rows, img.cols);
	sweepmatx = Mat(win, 180, CV_64FC1);
	sweepmaty = Mat(win, 180, CV_64FC1);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int w = 0; w < win; w++)
	{
		for (int deg = 0; deg < 180; deg++)
		{
			sweepmatx.at<double>(w, deg) = round((w + 1) * cos((deg + 1) * M_PI / 90));
			sweepmaty.at<double>(w, deg) = round((w + 1) * sin((deg + 1) * M_PI / 90));
		}
	}

	return 0;
}

vector<int> FindNearestPoints(const Point2d& point, const vector<Point2d>& crnrs, const int num, bool same)
{
	int size = crnrs.size();
	vector<pair<double, int>> list_dist(size);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int i = 0; i < size; i++)
		list_dist[i] = make_pair(pow(pow(point.x - crnrs[i].x, 2) + pow(point.y - crnrs[i].y, 2), 0.5), i);

	std::sort(list_dist.begin(), list_dist.end());

	vector<int> list_ind;
	int no = 0;
	for (int i = 0;i < list_dist.size();i++)
	{
		if (list_dist[i].first != 0)
		{
			list_ind.push_back(list_dist[i].second);
			no++;
			if (no == num)
			{
				return list_ind;
			}
		}
	}
	return vector<int>();
}

Point2d FindNearestPoints(const int index, const vector<Point2d>& crnrs)
{
	if (crnrs.size()<2)
	{
		return crnrs[0];
	}
	double dist, min_dist;
	int min_index;
	
	if (0 == index)
	{
		min_dist = pow(pow(crnrs[index].x - crnrs[1].x, 2) + pow(crnrs[index].y - crnrs[1].y, 2), 0.5);
		min_index = 1;
	}
	else
	{
		min_dist = pow(pow(crnrs[index].x - crnrs[0].x, 2) + pow(crnrs[index].y - crnrs[0].y, 2), 0.5);
		min_index = 0;
	}

	for (int i = 0; i < crnrs.size(); i++)
	{
		if (i == index)
			continue;

		dist = pow(pow(crnrs[index].x - crnrs[i].x, 2) + pow(crnrs[index].y - crnrs[i].y, 2), 0.5);
		if (min_dist > dist)
		{
			min_dist = dist;
			min_index = i;
		}
	}

	return crnrs[min_index];
}

int GetWin(const cv::Mat& img, const int index, const std::vector<cv::Point2d>& crnrs)
{
	Point2d nearestpt = FindNearestPoints(index, crnrs);

	const Point2d& p = crnrs[index];
	int win = (int)max(abs(nearestpt.x - p.x), abs(nearestpt.y - p.y)) - 2;

	// check for borders
	if (0 > p.x - win)
		win = p.x;

	if (p.x + win > img.cols - 1)
	{
		win = img.cols - 1 - p.x;
	}

	if (0 > p.y - win)
		win = p.y;

	if (p.y + win > img.rows - 1)
	{
		win = img.rows - 1 - p.y;
	}
	return win;
}

int AdjustImage(cv::Mat& img, const double* imeadjsca)
{
	double th;
	if (NULL == imeadjsca)
		th = 1;
	else
		th = *imeadjsca;

	Mat ming, stdv;
	meanStdDev(img, ming, stdv);

	double imax, imin;
	imax = ming.at<double>(0, 0) + th * stdv.at<double>(0, 0);
	imin = ming.at<double>(0, 0) - th * stdv.at<double>(0, 0);

	if (imax > 1)
		imax = 1;

	if (imin < 0)
		imin = 0;

#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++)
		{
			double& p = img.at<double>(r, c);
			p = (p - imin) / (imax - imin);
			p = p > 1 ? 1 : p;
			p = p < 0 ? 0 : p;
		}
	}

	return 0;
}

int CircleSweep(const cv::Mat& imgedge, const cv::Mat& sweepmatx, const cv::Mat& sweepmaty,
	cv::Mat& theta, cv::Mat& edgevalue, cv::Mat& thetasmd, cv::Mat& edgevaluesmd)
{
	int win = floor((double)imgedge.rows / 2.0);
	int cen = win + 1;

	Mat x, y;
	x = sweepmatx + cen;
	y = sweepmaty + cen;
	Mat values(x.rows, x.cols, CV_64FC1);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < values.rows; r++)
	{
		for (int c = 0; c < values.cols; c++)
		{
			double p_x = x.at<double>(r, c) - 1;
			double p_y = y.at<double>(r, c) - 1;
			double& p_v = values.at<double>(r, c);

			if (p_x < 0 || p_x >= imgedge.cols || p_y < 0 || p_y >= imgedge.rows)
				p_v = 0;
			else
				p_v = imgedge.at<double>(p_x, p_y);
		}
	}

	edgevalue = Mat(1, values.cols, CV_64FC1);
	for (int c = 0; c < values.cols; c++)
	{
		double sum_c = 0;
		for (int r = 0; r < values.rows; r++)
		{
			double p_v = values.at<double>(r, c);
			sum_c += p_v;
		}
		edgevalue.at<double>(0, c) = sum_c / values.rows;
	}

	edgevaluesmd = Mat(1, edgevalue.cols / 2, CV_64FC1);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int c = 0; c < edgevaluesmd.cols; c++)
	{
		edgevaluesmd.at<double>(0, c) = (edgevalue.at<double>(0, c) + edgevalue.at<double>(0, edgevalue.cols / 2 + c)) / 2;
	}

	return 0;
}

int Cadj(const Mat& edgevalue, vector<pair<int, double>>& v, double& v_max, double& v_min)
{
	Point min_p;
	minMaxLoc(edgevalue, &v_min, &v_max, &min_p);

	for (int i = min_p.x; i < edgevalue.cols; i++)
	{
		pair<int, double> p;
		p.first = i;
		p.second = edgevalue.at<double>(0, i);
		v.push_back(p);
	}

	for (int i = 0; i < min_p.x; i++)
	{
		pair<int, double> p;
		p.first = i;
		p.second = edgevalue.at<double>(0, i);
		v.push_back(p);
	}

	return 0;
}

int PeakDet(const cv::Mat& edgevalue, vector<pair<int, double>>& maxtab)
{
	vector<pair<int, double>> v;
	double v_min, v_max;
	Cadj(edgevalue, v, v_max, v_min);
	double delta = (v_max - v_min) / 4.0;

	if (0 == delta)
		return 1;

	double mn = 1e10;
	double mx = -1e10;
	int mxpos = v[0].first;

	bool lookformax = false;

	for (int i = 0; i < v.size(); i++)
	{
		pair<int, double> &p = v[i];

		if (p.second > mx)
		{
			mx = p.second;
			mxpos = p.first;
		}

		if (p.second < mn)
			mn = p.second;

		if (lookformax)
		{
			if (p.second < mx - delta)
			{
				maxtab.push_back(make_pair(mxpos, mx));
				mn = p.second;
				lookformax = false;
			}
		}
		else
		{
			if (p.second > mn + delta)
			{
				mx = p.second;
				mxpos = p.first;
				lookformax = true;
			}
		}
	}

	return 0;
}

double JudgeValidCorner(cv::Mat& imgcrop, cv::Mat& imgedgecrop, const cv::Mat& sweepmatxcrop, const cv::Mat& sweepmatycrop, vector<int>& peaklocs)
{
	// validcorner parameters
	double imadjsca = 0.8; // larger adjust scalars corresponds to less adjustment 0.8
	double imeadjsca = 1.8;  // 1.8
	double intth = 0.5;

	AdjustImage(imgedgecrop, &imeadjsca);

	Mat edgetheta, edgevalue, edgethetasmd, edgevaluesmd;
	CircleSweep(imgedgecrop, sweepmatxcrop, sweepmatycrop, edgetheta, edgevalue, edgethetasmd, edgevaluesmd);

	vector<pair<int, double>> maxtab;
	PeakDet(edgevalue, maxtab);

	bool valid;
	if (4 != maxtab.size())
	{
		valid = false;
		return -1;
	}

	for (int i = 0; i < maxtab.size(); i++)
	{
		peaklocs.push_back(maxtab[i].first);
	}

	vector<pair<int, double>> maxtabsmd;
	PeakDet(edgevaluesmd, maxtabsmd);

	if (2 != maxtabsmd.size())
	{
		valid = false;
		return -1;
	}

	AdjustImage(imgcrop, &imadjsca);
	Mat inttheta, intvalue, intthetasmd, intvaluesmd;
	CircleSweep(imgcrop, sweepmatxcrop, sweepmatycrop, inttheta, intvalue, intthetasmd, intvaluesmd);

	int loc1, loc2;
	if (maxtabsmd[0].first > maxtabsmd[1].first)
	{
		loc1 = maxtabsmd[1].first;
		loc2 = maxtabsmd[0].first;
	}
	else
	{
		loc1 = maxtabsmd[0].first;
		loc2 = maxtabsmd[1].first;
	}

	double crn1 = 0, crn2 = 0;
	int n = 0;
	for (int i = 0; i < loc1; i++)
	{
		crn1 += intvaluesmd.at<double>(0, i);
		n++;
	}
	for (int i = loc2; i < intvaluesmd.cols; i++)
	{
		crn1 += intvaluesmd.at<double>(0, i);
		n++;
	}
	crn1 = crn1 / n;

	n = 0;
	for (int i = loc1; i <= loc2; i++)
	{
		crn2 += intvaluesmd.at<double>(0, i);
		n++;
	}
	crn2 = crn2 / n;

	double result = abs(crn1 - crn2);
	return result;
}
double corr2(Mat matA, Mat matB) {
	//计算两个相同大小矩阵的二维相关系数  
	double corr2 = 0;

	double Amean2 = 0;
	double Bmean2 = 0;
	for (int m = 0; m < matA.rows; m++) {
		double* dataA = matA.ptr<double>(m);
		double* dataB = matB.ptr<double>(m);
		for (int n = 0; n < matA.cols; n++) {
			Amean2 = Amean2 + dataA[n];
			Bmean2 = Bmean2 + dataB[n];
		}
	}
	Amean2 = Amean2 / (matA.rows * matA.cols);
	Bmean2 = Bmean2 / (matB.rows * matB.cols);

	double Cov = 0;
	double Astd = 0;
	double Bstd = 0;
	for (int m = 0; m < matA.rows; m++) {
		double* dataA = matA.ptr<double>(m);
		double* dataB = matB.ptr<double>(m);
		for (int n = 0; n < matA.cols; n++) {
			//协方差  
			Cov = Cov + (dataA[n] - Amean2) * (dataB[n] - Bmean2);
			//A的方差  
			Astd = Astd + (dataA[n] - Amean2) * (dataA[n] - Amean2);
			//B的方差  
			Bstd = Bstd + (dataB[n] - Bmean2) * (dataB[n] - Bmean2);
		}
	}
	corr2 = Cov / (sqrt(Astd * Bstd));

	return corr2;
}
int LandMarkCornerFilter(const C_Pyramid& _pyr, const std::vector<cv::Point2d>& ctcrnrpts, std::vector<cv::Point2d>& crnrs, int &nocrnrs, std::vector<std::vector<int>>& peaklocs)
{
	const Mat& img = _pyr.m_MIm;
	const Mat& imgedge = _pyr.m_MIme;


	// set output values
	crnrs.clear();
	peaklocs.clear();

	nocrnrs = 0;

	vector<vector<int>> plocs;
	plocs.resize(ctcrnrpts.size());
	vector<double> valid_list(ctcrnrpts.size());
	// loop over all points

#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int i = 0; i < ctcrnrpts.size(); i++)
	{
		double x = ctcrnrpts[i].x;
		double y = ctcrnrpts[i].y;

		// extract appropriate window size
		int win = GetWin(img, i, ctcrnrpts);
		
		
		// check window size
		if (3 > win)
			continue;
		Mat imgcrop;
		img(Range(y - win, y + win + 1), Range(x - win, x + win + 1)).copyTo(imgcrop);
		Mat imgedgecrop;
		imgedge(Range(y - win, y + win + 1), Range(x - win, x + win + 1)).copyTo(imgedgecrop);

		AdjustImage(imgcrop);
		Mat templ(imgcrop.rows, imgcrop.cols, imgcrop.type(),0.0);
		templ(Rect(win+1,0,win,win)).setTo(1);
		templ(Rect(0, win+1, win, win)).setTo(1);
		templ.col(win).setTo(0.5);
		templ.row(win).setTo(0.5);

		Mat templ2(imgcrop.rows, imgcrop.cols, imgcrop.type(), 1.0);
		templ2(Rect(win + 1, 0, win, win)).setTo(0);
		templ2(Rect(0, win + 1, win, win)).setTo(0);
		templ2.col(win).setTo(0.5);
		templ2.row(win).setTo(0.5);

		double corr1_value = corr2(imgcrop, templ);
		double corr2_value = corr2(imgcrop, templ2);
		valid_list[i] = (corr1_value > corr2_value ? corr1_value : corr2_value);
	}
	double max_value = valid_list[0];
	int max_loc = 0;
	for (int i = 0; i < ctcrnrpts.size(); i++)
	{
		if (valid_list[i] > 0.8)            //协方差阈值设为0.8
		{
			crnrs.push_back(ctcrnrpts[i]);
			peaklocs.push_back(plocs[i]);
			nocrnrs++;

		}
		if (valid_list[i] > max_value)
		{
			max_value = valid_list[i];
			max_loc = i;
		}
	}
	if (nocrnrs==0)
	{
		crnrs.push_back(ctcrnrpts[max_loc]);
		peaklocs.push_back(plocs[max_loc]);
		nocrnrs++;
	}
	return 0;
}


int ChessCornerFilter(const C_Pyramid& _pyr, const std::vector<cv::Point2d>& ctcrnrpts, vector<Point2d>& crnrs, int &nocrnrs, vector<vector<int>>& peaklocs)
{
	const Mat& img = _pyr.m_MIm;
	const Mat& imgedge = _pyr.m_MIme;

	// get sweepmatrices, precalculation of these matrices allows for much
	// faster program execution
	Mat sweepmatx, sweepmaty;

	SweepMatrix(img, sweepmatx, sweepmaty);

	// set output values
	crnrs.clear();
	peaklocs.clear();

	nocrnrs = 0;

	vector<vector<int>> plocs;
	plocs.resize(ctcrnrpts.size());
	vector<double> valid_list(ctcrnrpts.size());
	// loop over all points

#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int i = 0; i < ctcrnrpts.size(); i++)
	{
		double x = ctcrnrpts[i].x;
		double y = ctcrnrpts[i].y;

		// extract appropriate window size
		int win = GetWin(img, i, ctcrnrpts);

		// check window size
		if (3 > win)
			continue;
		Mat imgcrop;
		img(Range(y - win, y + win + 1), Range(x - win, x + win + 1)).copyTo(imgcrop);
		Mat imgedgecrop;
		imgedge(Range(y - win, y + win + 1), Range(x - win, x + win + 1)).copyTo(imgedgecrop);
		const Mat sweepmatxcrop = sweepmatx(Range(0, (int)round(1.3*win)), Range::all());
		const Mat sweepmatycrop = sweepmaty(Range(0, (int)round(1.3*win)), Range::all());

		// 		// apply filter
		// 		vector<int> plocs; 
		// 		if( m_validcorner( imgcrop, imgedgecrop, sweepmatxcrop, sweepmatycrop, plocs ) )
		// 		{
		// 			crnrs.push_back( Point2d( x, y ) );
		// 			peaklocs.push_back( plocs );
		// 			nocrnrs++;
		// 		}
		valid_list[i] = JudgeValidCorner(imgcrop, imgedgecrop, sweepmatxcrop, sweepmatycrop, plocs[i]);
	}
	double max_value = valid_list[0];
	int max_loc = 0;
	for (int i = 0; i < ctcrnrpts.size(); i++)
	{
		if (valid_list[i] > 0.5)
		{
			crnrs.push_back(ctcrnrpts[i]);
			peaklocs.push_back(plocs[i]);
			nocrnrs++;
		}
		if (valid_list[i]>max_value)
		{
			max_value = valid_list[i];
			max_loc = i;
		}
	}
	if (nocrnrs==0)
	{
		crnrs.push_back(ctcrnrpts[max_loc]);
		peaklocs.push_back(plocs[max_loc]);
		nocrnrs++;
	}
	return 0;
}

std::vector<cv::Point2d> GetControlPoints(const cv::Mat& _imgh, const cv::Mat& _mimg, const cv::Mat& _stdv, const double& _th)
{
	// adaptive thresholding
	Mat imax;
	addWeighted(_mimg, 1.0, _stdv, _th, 0, imax);
#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int r = 0; r < imax.rows; r++)
	{
		for (int c = 0; c < imax.cols; c++)
		{
			double& p = imax.at<double>(r, c);
			p = p > 1.0 ? 1.0 : p;
			p = p < 0.0 ? 0.0 : p;
		}
	}

	Mat imghl = Mat::zeros(imax.rows, imax.cols, CV_8UC1);
	compare(_imgh, imax, imghl, CMP_GT);

	Mat imghlf = Mat::zeros(imax.rows, imax.cols, CV_8UC1);
	medianBlur(imghl, imghlf, 3);
	medianBlur(imghlf, imghlf, 3);

	// get centroids of blobs as Harris corner points
	vector<vector<Point>> contours;
	findContours(imghlf, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	std::vector<cv::Point2d> ctcrnrpts(contours.size());
	Mat crnrpts = Mat::zeros(2, contours.size(), CV_64FC1);

#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
	for (int i = 0; i < contours.size(); i++)
	{
		Point2d crnrpt;
		vector<Point>& Points = contours[i];
		for (int j = 0; j < Points.size(); j++)
		{
			crnrpt.x += Points[j].x;
			crnrpt.y += Points[j].y;
		}
		crnrpt.x = round(crnrpt.x / Points.size());
		crnrpt.y = round(crnrpt.y / Points.size());
		ctcrnrpts[i] = crnrpt;
	}
	return ctcrnrpts;
}

double Deg2Rad(const double angdegs)
{
	return M_PI * angdegs / 180;
}

bool AngleProx(const double ang1, const double ang2, const double th)
{
	return abs(ang1 - ang2) < th || abs(ang1 - ang2) > (2 * M_PI - th);
}

int GetGrid(const std::vector<cv::Point2d>& crnrs, const std::vector<cv::Point2d>& crnrpts, const vector<vector<int>>& peaklocs, const C_Pyramid& _pyr, cv::Mat& crnrsgridout)
{
	int nocrnrs = (int)crnrs.size();
	const Mat& ix = _pyr.m_MIx;
	const Mat& iy = _pyr.m_MIy;

	// get mean point
	Point2d centerpt;
	for (int i = 0; i < nocrnrs; i++)
	{
		centerpt.x += crnrs[i].x;
		centerpt.y += crnrs[i].y;
	}
	centerpt.x /= nocrnrs;
	centerpt.y /= nocrnrs;
	int ctindx = FindNearestPoints(centerpt, crnrs)[0];

	// parameters
	double angth = Deg2Rad(15);

	// always store first point
	bool valid = true;

	// inititalise matrices
	Mat crnrsgrid = Mat::zeros(nocrnrs, nocrnrs, CV_64FC1);
	Mat crnrsgriddone = Mat::zeros(nocrnrs, nocrnrs, CV_64FC1);

	// place first point in the middle of the grid
	int xg = round(nocrnrs / 2) - 1;
	int yg = round(nocrnrs / 2) - 1;

	// enumerate different directions
	int right = 1;
	int top = 2;
	int left = 3;
	int bottom = 4;

	// setup position matrix 
	Mat posmat = (Mat_<double>(3, 3) << 0, top, 0, left, 0, right, 0, bottom, 0);

	// set while loop flag
	bool notdone = true;

	// set loop counter
	// just to ensure safe performance, prevent loop from going on forever
	int loopcntr = 0;
	int looplimit = 1e6;

	while (notdone && loopcntr < looplimit)
	{
		loopcntr++;
		// get current point coords
		Point2d currentpt = crnrs[ctindx];
		int xc = currentpt.x;
		int yc = currentpt.y;

		// get surrounding chessboard corner properties (sorted by distance)
		vector<int> surindx = FindNearestPoints(currentpt, crnrs, 8);
		vector<double> angles;
		angles.resize(surindx.size());
		for (int i = 0; i < angles.size(); i++)
		{
			int x = crnrs[surindx[i]].x - currentpt.x;
			int y = crnrs[surindx[i]].y - currentpt.y;
			angles[i] = atan2(y, x);
			angles[i] = angles[i] <= 0 ? angles[i] + 2 * M_PI : angles[i];
		}

		// setup the segment vectors to the corners in order to check for 
		// validity of segment between points by summing ix and iy along segment

		vector<Point2d> vecsnrm; // normal vectors
		vecsnrm.resize(surindx.size(), Point2d(0, 0));
		vector<double> vecslen; // vector lengths
		vecslen.resize(surindx.size(), 0);
		for (int veccnr = 0; veccnr < surindx.size(); veccnr++)
		{
			int x = crnrs[surindx[veccnr]].x - xc;
			int y = crnrs[surindx[veccnr]].y - yc;
			vecslen[veccnr] = pow(std::norm(complex<double>(x, y)), 0.5);
			vecsnrm[veccnr].x = x / vecslen[veccnr];
			vecsnrm[veccnr].y = y / vecslen[veccnr];
		}

		vector<double> ixvalue; // ix values
		ixvalue.resize(surindx.size(), 0);
		vector<double> iyvalue; // iy values
		iyvalue.resize(surindx.size(), 0);
		vector<double> segedgevalue;
		segedgevalue.resize(surindx.size());

		for (int crnrcnr = 0; crnrcnr < surindx.size(); crnrcnr++)
		{
			for (int evcnr = 1; evcnr <= round(vecslen[crnrcnr]); evcnr++)
			{
				int xev = round(xc + vecsnrm[crnrcnr].x * evcnr);
				int yev = round(yc + vecsnrm[crnrcnr].y * evcnr);
				ixvalue[crnrcnr] += ix.at<double>(xev, yev);
				iyvalue[crnrcnr] += iy.at<double>(xev, yev);
			}
			segedgevalue[crnrcnr] =
				pow(std::norm(complex<double>(
					ixvalue[crnrcnr] / round(vecslen[crnrcnr]),
					iyvalue[crnrcnr] / round(vecslen[crnrcnr]))), 0.5);
		}
		double segedgemean = 0;
		for (int i = 0; i < segedgevalue.size(); i++)
		{
			segedgemean += segedgevalue[i];
		}
		segedgemean /= segedgevalue.size();

		// get surrounding Harris point properties (sorted by distance)
		vector<int> surindex = FindNearestPoints(currentpt, crnrpts, 8);
		vector<double> anglespixs;
		anglespixs.resize(surindex.size());
		for (int i = 0; i < anglespixs.size(); i++)
		{
			int x = crnrpts[surindex[i]].x - currentpt.x;
			int y = crnrpts[surindex[i]].y - currentpt.y;
			anglespixs[i] = atan2(y, x);
			anglespixs[i] = anglespixs[i] <= 0 ? anglespixs[i] + 2 * M_PI : anglespixs[i];
		}

		vector<int> locs = peaklocs[ctindx];

		if (4 != locs.size())
			cerr << "There should be 4 and only 4 peaks" << endl;

		vector<double> lineangles;
		for (int i = 0; i < locs.size(); i++)
		{
			lineangles.push_back((double)(locs[i] + 1) * M_PI / 90);
		}

		// get cross corners

		// reset crosspixs
		vector<double> crosspixs;
		crosspixs.resize(4, 0);

		for (int pk = 0; pk < locs.size(); pk++)
		{
			for (int crnr = 0; crnr < surindx.size(); crnr++)
			{
				// check for angle proximity and segment edge projection
				if (AngleProx(angles[crnr], lineangles[pk], angth) &&
					segedgevalue[crnr] > segedgemean);
				{
					for (int pix = 0; pix < surindex.size(); pix++)
					{
						// check if a Harris corner lies in between
						if (AngleProx(anglespixs[pix], lineangles[pk], angth))
						{
							if (crnrpts[surindex[pix]].x == crnrs[surindx[crnr]].x &&
								crnrpts[surindex[pix]].y == crnrs[surindx[crnr]].y)
							{
								// store
								crosspixs[pk] = surindx[crnr];
								break;
							}
							else
								break;
						}
					}
				}
			}
		}

		// Adjust cross
		for (int i = 0; i < crosspixs.size(); i++)
		{
			for (int u = xg - 1; u <= xg + 1; u++)
			{
				for (int v = yg - 1; v <= yg + 1; v++)
				{
					if (crosspixs[i] == crnrsgrid.at<double>(u, v) &&
						crnrsgrid.at<double>(u, v) > 0)
					{
						valid = 1; // a cross is valid if a match is found
						int k = (int)posmat.at<double>(u - xg + 1, v - yg + 1) - i - 1;
						vector<double> temp(4);
						for (int m = 1;m <= 4;m++)
							temp[m - 1] = crosspixs[((m - k + 7)) % 4];
						crosspixs = temp;
					}
				}
			}
		}

		if (valid)
		{
			Mat cmat = cv::Mat::zeros(3, 3, CV_64FC1);
			cmat.at<double>(1, 1) = ctindx;
			cmat.at<double>(1, 2) = crosspixs[0];
			cmat.at<double>(0, 1) = crosspixs[1];
			cmat.at<double>(1, 0) = crosspixs[2];
			cmat.at<double>(2, 1) = crosspixs[3];

			Mat crnrsgrid_sub(crnrsgrid, Rect(yg - 1, xg - 1, 3, 3));
			Mat crnrsgriddone_sub(crnrsgriddone, Rect(yg - 1, xg - 1, 3, 3));
			Mat not_crnrsgrid_sub, bit_and;
			MatElementZero(crnrsgrid_sub, not_crnrsgrid_sub);
			crnrsgrid_sub = crnrsgrid_sub + cmat.mul(not_crnrsgrid_sub);
			Mat cmatdone = (Mat_<double>(3, 3) << 0, 1, 0, 1, 2, 1, 0, 1, 0);
			MatElementBitAnd(cmatdone, cmat, bit_and);
			cmatdone = cmatdone.mul(bit_and) - crnrsgriddone_sub;
			for (int i = 0;i < 3;i++)
				for (int j = 0;j < 3;j++)
					if (cmatdone.at<double>(i, j) < 0)
						cmatdone.at<double>(i, j) = 0;
			crnrsgriddone_sub = cmatdone + crnrsgriddone_sub;
			valid = 0;
		}
		else
		{
			crnrsgriddone.at<double>(xg, yg) = 0;
		}
		int width = crnrsgriddone.cols;
		int height = crnrsgriddone.rows;
		bool find_1 = false;
		for (int i = 0;i < width&&!find_1;i++)
		{
			for (int j = 0;j < height&&!find_1;j++)
			{
				if (crnrsgriddone.at<double>(j, i) == 1)
				{
					find_1 = true;
					xg = j;
					yg = i;
				}
			}
		}
		if (!find_1)
			notdone = 0;
		else
			ctindx = crnrsgrid.at<double>(xg, yg);
	}
	int width = crnrsgrid.cols;
	int height = crnrsgrid.rows;
	crnrsgridout.create(height, width, CV_64FC2);
	crnrsgridout.setTo(0.0);
	for (int i = 0;i < height;i++)
	{
		for (int j = 0;j < width;j++)
		{
			double &vgrid = crnrsgrid.at<double>(i, j);
			if (vgrid)
				crnrsgridout.at<cv::Vec2d>(i, j) = cv::Vec2d(crnrs[(int)vgrid].y, crnrs[(int)vgrid].x);
		}
	}
	return 0;
}

int FilterGrid(const cv::Mat& crnrsgrid, cv::Mat& crnrsgridfil)
{
	crnrsgrid.copyTo(crnrsgridfil);
	while (true)
	{
		vector<Mat> mats;
		cv::split(crnrsgridfil, mats);

		double cols = mats[0].cols, rows = mats[0].rows;
		Mat first_row = mats[0].row(0);
		Mat last_row = mats[0].row(rows - 1);
		Mat first_col = mats[0].col(0);
		Mat last_col = mats[0].col(cols - 1);

		vector<pair<double, int>> list;

		list.push_back(make_pair(cv::countNonZero(first_row) / cols, 0));
		list.push_back(make_pair(cv::countNonZero(last_row) / cols, 1));
		list.push_back(make_pair(cv::countNonZero(first_col) / rows, 2));
		list.push_back(make_pair(cv::countNonZero(last_col) / rows, 3));

		std::sort(list.begin(), list.end());

		if (list[0].first < 0.5)
		{
			int index = list[0].second;
			if (index == 0)
			{
				Mat temp(crnrsgridfil, Rect(0, 1, cols, rows - 1));
				temp.copyTo(crnrsgridfil);
			}
			else if (index == 1)
			{
				Mat temp(crnrsgridfil, Rect(0, 0, cols, rows - 1));
				temp.copyTo(crnrsgridfil);

			}
			else if (index == 2)
			{
				Mat temp(crnrsgridfil, Rect(1, 0, cols - 1, rows));
				temp.copyTo(crnrsgridfil);

			}
			else if (index == 3)
			{
				Mat temp(crnrsgridfil, Rect(0, 0, cols - 1, rows));
				temp.copyTo(crnrsgridfil);
			}
		}
		else
			break;

	}

	return 0;
}

double point2Line(Point2f p1, Point2f lp1, Point2f lp2)
{
	double a, b, c, dis;
	// 化简两点式为一般式
	// 两点式公式为(y - y1)/(x - x1) = (y2 - y1)/ (x2 - x1)
	// 化简为一般式为(y2 - y1)x + (x1 - x2)y + (x2y1 - x1y2) = 0
	// A = y2 - y1
	// B = x1 - x2
	// C = x2y1 - x1y2
	a = lp2.y - lp1.y;
	b = lp1.x - lp2.x;
	c = lp2.x * lp1.y - lp1.x * lp2.y;
	// 距离公式为d = |A*x0 + B*y0 + C|/√(A^2 + B^2)
	dis = abs(a * p1.x + b * p1.y + c) / sqrt(a * a + b * b);
	return dis;
};
double point2point(Point2f p1, Point2f p2)
{
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}
int CornerFinder::AdjustCornerDirection(std::vector<cv::Point2f> &pts,const cv::Size &pattern_size, int origin_pos)
{
	double grid_length = point2point(pts[0], pts[1]);
	int min_size = std::min(pattern_size.width, pattern_size.height);
	int w,h;
	if (point2point(pts[min_size], pts[0]) > grid_length * 2)
	{
		w = pattern_size.width;
		h = pattern_size.height;
	}
	else
	{
		w = pattern_size.height;
		h = pattern_size.width;
	}
	cv::Mat crnrsgridfil(h, w, CV_64FC2);
	for (size_t i = 0; i < h; i++)
	{
		for (size_t j = 0; j < w; j++)
		{
			const cv::Point2f &p = pts[i*w+j];
			crnrsgridfil.at<Vec2d>(i,j) = Vec2d(p.x, p.y);
		}
	}
	AdjustGridDirection(crnrsgridfil);
	AdjustGridOrigin(crnrsgridfil, origin_pos);

	if (crnrsgridfil.cols != pattern_size.width)
	{
		transpose(crnrsgridfil, crnrsgridfil);
	}


	pts.clear();
	h = pattern_size.height;
	w = pattern_size.width;
	for (size_t i = 0; i < h; i++)
	{
		for (size_t j = 0; j < w; j++)
		{
			const Vec2d &p = crnrsgridfil.at<Vec2d>(i,j);
			pts.push_back(Point2f(p[0], p[1]));
		}
	}
	return 0;
}
int AdjustGridDirection(cv::Mat& crnrsgridfil)
{
	int rows = crnrsgridfil.rows, cols = crnrsgridfil.cols;
	double row_avg = 0, col_avg = 0;
	vector<Mat> grids;
	cv::split(crnrsgridfil, grids);
	for (int rowindx = 0;rowindx < rows;rowindx++)
	{
		Mat currentrowx = grids[0].row(rowindx);
		Mat currentrowy = grids[1].row(rowindx);
		vector<Point2d> points;
		double tempx, tempy;
		for (int i = 0;i < cols;i++)
		{
			tempx = currentrowx.at<double>(0, i);
			tempy = currentrowy.at<double>(0, i);
			if (tempx > 0 && tempy > 0)
				points.push_back(Point2d(tempx, tempy));
		}
		cv::Vec4d line;
		cv::fitLine(points, line, CV_DIST_L2, 0, 0.001, 0.001);
		row_avg += abs(line[1] / line[0]);
	}
	row_avg /= rows;
	for (int colindx = 0;colindx < cols;colindx++)
	{
		Mat currentcolx = grids[0].col(colindx);
		Mat currentcoly = grids[1].col(colindx);
		vector<Point2d> points;
		double tempx, tempy;
		for (int i = 0;i < rows;i++)
		{
			tempx = currentcolx.at<double>(i, 0);
			tempy = currentcoly.at<double>(i, 0);
			if (tempx > 0 && tempy > 0)
				points.push_back(Point2d(tempx, tempy));
		}
		cv::Vec4d line;
		cv::fitLine(points, line, CV_DIST_L2, 0, 0.001, 0.001);
		col_avg += abs(line[1] / line[0]);
	}
	col_avg /= cols;

	if (col_avg > row_avg)
	{
		cv::transpose(crnrsgridfil, crnrsgridfil);
	}

	return 0;
}

bool Mat2Points(const cv::Mat &matx, const cv::Mat &maty, std::vector<cv::Point2d> &points, bool is_row)
{
	if (matx.type() != CV_64FC1 || maty.type() != CV_64FC1)
	{
		return false;
	}
	points.clear();
	int length = is_row ? matx.cols : matx.rows;
	double tempx, tempy;
	if (is_row)
	{
		for (size_t i = 0; i < length; i++)
		{
			tempx = matx.at<double>(0, i);
			tempy = maty.at<double>(0, i);
			if (tempx > 0 && tempy > 0)
				points.push_back(Point2d(tempx, tempy));
		}
	}
	else
	{
		for (size_t i = 0; i < length; i++)
		{
			tempx = matx.at<double>(i, 0);
			tempy = maty.at<double>(i, 0);
			if (tempx > 0 && tempy > 0)
				points.push_back(Point2d(tempx, tempy));
		}

	}

	return true;
}

int FixMissCorners(const cv::Mat& crnrsgridfil, const cv::Size img_size, cv::Mat& outgrid, int& nointerpolations)
{
	vector<Mat> grids;
	split(crnrsgridfil, grids);
	nointerpolations = 0;
	int rows = crnrsgridfil.rows, cols = crnrsgridfil.cols;
	// list of interpolations that lie outside the image
	outgrid.create(rows, cols, CV_64FC1);
	outgrid.setTo(0.0);

	for (size_t x = 0; x < rows; x++)
	{
		for (size_t y = 0; y < cols; y++)
		{
			if (grids[0].at<double>(x, y) == 0)
			{
				vector<Point2d> wy_wx, lx_ly;
				Mat2Points(grids[1].row(x), grids[0].row(x), wy_wx, true);
				Mat2Points(grids[0].col(y), grids[1].col(y), lx_ly, false);
				if (wy_wx.size() < 2 || lx_ly.size() < 2)
				{
					outgrid.release();
					return -1;
				}
				nointerpolations++;
				double k1, k2, d1, d2;
				FitLine2D(wy_wx, k1, d1);
				FitLine2D(lx_ly, k2, d2);
				Mat eqnsmat = (Mat_<double>(2, 2) << 1, -k1, -k2, 1);
				Mat bmat = (Mat_<double>(2, 1) << d1, d2);
				Mat result = eqnsmat.inv()*bmat;
				grids[0].at<double>(x, y) = result.at<double>(0, 0);
				grids[1].at<double>(x, y) = result.at<double>(1, 0);
				if (grids[0].at<double>(x, y) < 0 || grids[0].at<double>(x, y) >= img_size.width || grids[1].at<double>(x, y) < 0 || grids[1].at<double>(x, y) >= img_size.height)
				{
					outgrid.at<double>(x, y) = 1;
				}
			}
		}
	}
	while (countNonZero(outgrid) != 0)
	{
		int r, c;
		if (outgrid.at<double>(0, 0))
		{
			r = c = 0;
		}
		else if (outgrid.at<double>(0, cols - 1))
		{
			r = 0;
			c = cols - 1;
		}
		else if (outgrid.at<double>(rows - 1, 0))
		{
			r = rows - 1;
			c = 1;
		}
		else if (outgrid.at<double>(rows - 1, cols - 1))
		{
			r = rows - 1;
			c = cols - 1;
		}
		else
		{
			outgrid.release();
			return -1;
		}
		if (cols > rows)
		{
			MatDeleteOneCol(outgrid, c);
			MatDeleteOneCol(grids[0], c);
			MatDeleteOneCol(grids[1], c);
		}
		else
		{
			MatDeleteOneRow(outgrid, r);
			MatDeleteOneRow(grids[0], r);
			MatDeleteOneRow(grids[1], r);
		}
	}
	cv::merge(grids, outgrid);
	return 0;
}

int FitLine2D(const std::vector<cv::Point2d> &points, double &k, double &d)
{
	cv::Vec4d line;
	cv::fitLine(points, line, CV_DIST_L2, 0, 0.001, 0.001);
	k = line[1] / line[0];
	d = line[3] - k * line[2];
	return 0;
}

int AdjustGridOrigin(cv::Mat& gridfullrect,int origin_pos)
{
	int rows = gridfullrect.rows, cols = gridfullrect.cols;
	Vec2d v1 = gridfullrect.at<Vec2d>(0, 0);
	Vec2d v2 = gridfullrect.at<Vec2d>(rows - 1, cols - 1);
	if (v1[0] < v2[0] && v1[1] < v2[1])
	{
	}
	else if (v1[0]<v2[0] && v1[1]>v2[1])
	{
		flip(gridfullrect, gridfullrect, 1);
	}
	else if (v1[0] > v2[0] && v1[1] < v2[1])
	{
		flip(gridfullrect, gridfullrect, 0);
	}
	else if (v1[0] > v2[0] && v1[1] > v2[1])
	{
		flip(gridfullrect, gridfullrect, -1);
	}
	switch (origin_pos)
	{
	case 1:
		flip(gridfullrect, gridfullrect, 0);
		break;
	case 2:
		flip(gridfullrect, gridfullrect, 1);
		break;
	case 3:
		flip(gridfullrect, gridfullrect, -1);
		break;
	default:
		break;
	}
	return 0;
}

void InvPointsXY(std::vector<cv::Point2d> &out)
{
	int size = out.size();
	std::vector<cv::Point2d> temp;
	for (int i = 0;i < size;i++)
	{
		const cv::Point2d &p = out[i];
		temp.push_back(cv::Point2d(p.y, p.x));
	}
	out = temp;
}

int CornerFinder::FindCornersOfAutoFixMissingCorners(const cv::Mat &img, std::vector<cv::Point2f> &corners,cv::Mat &out /*= cv::Mat()*/)
{
	cv::Mat input_img;
	img.copyTo(input_img);
	cv::Size corner_size;
	corners.clear();

	Mat I_64FC1(input_img.rows, input_img.cols, CV_64FC1);

	if (input_img.type() == CV_8UC3)
	{

#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
		for (int r = 0; r < input_img.rows; r++)
		{
			for (int c = 0; c < input_img.cols; c++)
			{
				Vec3b& bgr = input_img.at<Vec3b>(r, c);
				double& v = I_64FC1.at<double>(r, c);
				double B = (int)bgr[0];
				double G = (int)bgr[1];
				double R = (int)bgr[2];
				v = 0.299 * R + 0.5870 * G + 0.114 * B;
			}
		}
	}
	else if (input_img.type() == CV_8UC1)
	{
		input_img.convertTo(I_64FC1, CV_64FC1);
	}
	else
		return -1;

	if (50 > I_64FC1.rows || 50 > I_64FC1.cols)
	{
		cerr << "Image too small" << endl;
		return 1;
	}

	// change image into 0~1
	GlobalScale(I_64FC1);

	// set nocrnrs and level
	int nocrnrs = -1;
	vector<Point2d> crnrs;
	vector<Point2d> crnrpts;
	vector<vector<int>> peaklocs;
	int level = -1;

	// Make Pyramid
	for (int cntr = 0; cntr < m_iNoLevels; cntr++)
	{
		m_pPyr[cntr].m_scale = pow(2, cntr);
		// downsize image
		Mat& im = m_pPyr[cntr].m_MIm;
		Size& im_size = Size(I_64FC1.cols / m_pPyr[cntr].m_scale, I_64FC1.rows / m_pPyr[cntr].m_scale);
		cv::resize(I_64FC1, im, im_size, INTER_LINEAR);

		// adaptively adjust
		Mat& imadj = m_pPyr[cntr].m_MImadj;
		AdaptImageAdjust(im, imadj);

		// get sobel edge image
		Mat& imgedge = m_pPyr[cntr].m_MIme;
		Mat& ix = m_pPyr[cntr].m_MIx;
		Mat& iy = m_pPyr[cntr].m_MIy;
		GetEdges(im, imgedge, ix, iy);

		Mat imgh, mimg, stdv;
		HarrisCorners(m_pPyr[cntr].m_MImadj, imgh, &m_iHWin);

		AdaptStatus(imgh, mimg, stdv);

		m_pPyr[cntr].ctcrnrpts = GetControlPoints(imgh, mimg, stdv, m_dTh);

		if (0 == m_pPyr[cntr].ctcrnrpts.size())
			continue;

		ChessCornerFilter(m_pPyr[cntr], m_pPyr[cntr].ctcrnrpts, m_pPyr[cntr].ctcrnrs, m_pPyr[cntr].ctnocrnrs, m_pPyr[cntr].ctpeaklocs);

		if (m_pPyr[cntr].ctnocrnrs > nocrnrs)
		{
			nocrnrs = m_pPyr[cntr].ctnocrnrs;
			crnrs = m_pPyr[cntr].ctcrnrs;
			crnrpts = m_pPyr[cntr].ctcrnrpts;
			peaklocs = m_pPyr[cntr].ctpeaklocs;
			level = cntr;
		}

		// 		Mat img_test;
		// 		cv::normalize(m_pPyr[cntr].m_MIm, img_test, 255, 0, CV_MINMAX);
		// 		img_test.convertTo(img_test, CV_8UC1);
		// 		cvtColor(img_test, img_test, CV_GRAY2BGR);
		// 		for (int i = 0;i < crnrs.size();i++)
		// 		{
		// 			circle(img_test, crnrs[i], 1, CV_RGB(255, 0, 0));
		// 		}
		// 		cvNamedWindow("test", CV_WINDOW_KEEPRATIO);
		// 		imshow("test", img_test);
		// 		cvWaitKey();

	}

	// Check for enough no of corners
	if (10 > nocrnrs)
	{
		cout << "No of corners too small" << endl;
		return 1;
	}

	InvPointsXY(crnrs);
	InvPointsXY(crnrpts);

	// Extract Grid
	Mat crnrsgrid;
	GetGrid(crnrs, crnrpts, peaklocs, m_pPyr[level], crnrsgrid);

	// adjust grid back to full scale
	crnrsgrid = crnrsgrid*(pow(2.0, level));
	for (int i = 0;i < crnrpts.size();i++)
		crnrpts[i] = crnrpts[i] * (pow(2.0, level));

	Mat crnrsgridfil;
	FilterGrid(crnrsgrid, crnrsgridfil);

	// check grid size
	if (3 > crnrsgridfil.rows || 3 > crnrsgridfil.cols)
	{
		return 1;
	}



	// get missing corners
	Mat gridfullrect;
	int nointerpolations;
	FixMissCorners(crnrsgridfil, Size(I_64FC1.cols, I_64FC1.rows), gridfullrect, nointerpolations);

	// adjust grid direction
	AdjustGridDirection(gridfullrect);

	if (gridfullrect.empty())
	{
		return 1;
	}

	// 	// adjust origin position
	AdjustGridOrigin(gridfullrect);

	vector<Point2d> corners_d;
	Size patten_size(gridfullrect.cols, gridfullrect.rows);
	for (size_t i = 0; i < patten_size.height; i++)
	{
		for (size_t j = 0; j < patten_size.width; j++)
		{
			Vec2d &v = gridfullrect.at<Vec2d>(i, j);
			corners.push_back(Point2f(v[0], v[1]));
			corners_d.push_back(Point2d(v[0], v[1]));

		}
	}
	int win_radius = 10000000, ct_win;
	for (size_t i = 0; i < corners.size(); i++)
	{
		ct_win = round(GetWin(I_64FC1, i, corners_d) / 2.0);

		if (ct_win < win_radius&&ct_win>2)
		{
			win_radius = ct_win;
		}

	}

	cv::normalize(I_64FC1, I_64FC1, 255, 0, CV_MINMAX);
	I_64FC1.convertTo(I_64FC1, CV_8UC1);
	cornerSubPix(I_64FC1, corners, Size(win_radius, win_radius), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	corner_size = patten_size;
	cvtColor(I_64FC1, out, CV_GRAY2BGR);
	drawChessboardCorners(out, patten_size, corners, true);

	if (corners.size() == corner_size.width*corner_size.height)
	{
		return 0;
	}
	return -1;
}

std::vector<cv::Point2f> CornerFinder::FindSeparateCorners(const cv::Mat &img, cv::Mat &out /*= cv::Mat()*/)
{
	cv::Mat input_img = img;
	cv::Size corner_size;
	std::vector<cv::Point2f> corners;

	Mat I_64FC1(input_img.rows, input_img.cols, CV_64FC1);

	if (input_img.type() == CV_8UC3)
	{

#if MULT_THREAD
#pragma omp parallel for
#endif // MULT_THREAD
		for (int r = 0; r < input_img.rows; r++)
		{
			for (int c = 0; c < input_img.cols; c++)
			{
				Vec3b& bgr = input_img.at<Vec3b>(r, c);
				double& v = I_64FC1.at<double>(r, c);
				double B = (int)bgr[0];
				double G = (int)bgr[1];
				double R = (int)bgr[2];
				v = 0.299 * R + 0.5870 * G + 0.114 * B;
			}
		}
	}
	else if (input_img.type() == CV_8UC1)
	{
		input_img.convertTo(I_64FC1, CV_64FC1);
	}
	else
		return corners;

	// change image into 0~1
	GlobalScale(I_64FC1);

	// set nocrnrs and level
	int nocrnrs = -1;
	vector<Point2d> crnrs;
	vector<Point2d> crnrpts;
	vector<vector<int>> peaklocs;
	int level = -1;

	// Make Pyramid
	for (int cntr = 0; cntr < m_iNoLevels; cntr++)
	{
		m_pPyr[cntr].m_scale = pow(2, cntr);
		// downsize image
		Mat& im = m_pPyr[cntr].m_MIm;
		Size& im_size = Size(I_64FC1.cols / m_pPyr[cntr].m_scale, I_64FC1.rows / m_pPyr[cntr].m_scale);
		cv::resize(I_64FC1, im, im_size, INTER_LINEAR);

		// adaptively adjust
		Mat& imadj = m_pPyr[cntr].m_MImadj;
		AdaptImageAdjust(im, imadj);

		// get sobel edge image
		Mat& imgedge = m_pPyr[cntr].m_MIme;
		Mat& ix = m_pPyr[cntr].m_MIx;
		Mat& iy = m_pPyr[cntr].m_MIy;
		GetEdges(im, imgedge, ix, iy);

		Mat imgh, mimg, stdv;
		HarrisCorners(m_pPyr[cntr].m_MImadj, imgh, &m_iHWin);

		AdaptStatus(imgh, mimg, stdv);

		m_pPyr[cntr].ctcrnrpts = GetControlPoints(imgh, mimg, stdv, m_dTh);

		if (0 == m_pPyr[cntr].ctcrnrpts.size())
			continue;

		LandMarkCornerFilter(m_pPyr[cntr], m_pPyr[cntr].ctcrnrpts, m_pPyr[cntr].ctcrnrs, m_pPyr[cntr].ctnocrnrs, m_pPyr[cntr].ctpeaklocs);

		if (m_pPyr[cntr].ctnocrnrs > nocrnrs)
		{
			nocrnrs = m_pPyr[cntr].ctnocrnrs;
			crnrs = m_pPyr[cntr].ctcrnrs;
			crnrpts = m_pPyr[cntr].ctcrnrpts;
			peaklocs = m_pPyr[cntr].ctpeaklocs;
			level = cntr;
		}

// 		Mat img_test;
// 		cv::normalize(m_pPyr[cntr].m_MIm, img_test, 255, 0, CV_MINMAX);
// 		img_test.convertTo(img_test, CV_8UC1);
// 		cvtColor(img_test, img_test, CV_GRAY2BGR);
// 		for (int i = 0; i < crnrs.size(); i++)
// 		{
// 			circle(img_test, crnrs[i], 5, CV_RGB(255, 0, 0));
// 		}
// 		cout << endl;

	}

	int win_radius = 3;
	for (size_t i = 0; i < crnrs.size(); i++)
	{
		corners.push_back(Point2f(crnrs[i].x, crnrs[i].y));
	}
	if (corners.size() > 0)
	{
		cv::normalize(I_64FC1, I_64FC1, 255, 0, CV_MINMAX);
		I_64FC1.convertTo(I_64FC1, CV_8UC1);
		cornerSubPix(I_64FC1, corners, Size(win_radius, win_radius), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cvtColor(I_64FC1, out, CV_GRAY2BGR);
		for (int i = 0; i < corners.size(); i++)
			circle(out, corners[i], 3, CV_RGB(255, 0, 0));
	}
	return corners;
}
vector<cv::Point2f> CornerFinder::FindSubPixCorners(const cv::Mat &img, const std::vector<cv::Point2f>& pts,int win_radius) const
{
	Mat temp;
	if (img.channels() == 3)
		cvtColor(img, temp, CV_BGR2GRAY);
	else
		img.copyTo(temp);
	vector<Point2f> corners = pts;
	cornerSubPix(temp, corners, Size(win_radius, win_radius), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	return corners;
}
