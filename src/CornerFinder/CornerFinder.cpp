#include <iostream>
#include <vector>
#include "CornerFinder.h"
#include "Common_GQ.h"
#include <opencv2/ccalib/omnidir.hpp>
using namespace Common_GQ;
using namespace std;
using namespace cv;

CornerFinder::CornerFinder()
{
	m_iHWin = 3;
	m_dTh = 0.5;
	m_iNoLevels = 3;
	m_pPyr = new C_Pyramid[m_iNoLevels];
}

CornerFinder::~CornerFinder()
{
	delete [] m_pPyr;
	m_pPyr = NULL;
}


void CornerFinder::CalcChessboardCorners(Size boardSize, float squareSize, int nums, vector<vector<Point3f>> & object_Points)
{
	vector<Point3f> corners;
	corners.resize(0);
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(float(j*squareSize),
				float(i*squareSize), 0));
	object_Points.resize(nums, corners);
}


bool CornerFinder::FindCornersOfOpenCV(const cv::Mat &img, std::vector<cv::Point2f> &corners,const cv::Size& pattern_size, int sub_radius, cv::Mat &out /*= cv::Mat()*/)
{
	cv::Mat gray_img;
	if (img.channels() == 3)
	{
		cvtColor(img, gray_img, CV_BGR2GRAY);
	}
	else if (img.channels() == 1)
		img.copyTo(gray_img);
	else
		return false;
	corners.clear();
	bool patternfound = cv::findChessboardCorners(gray_img, pattern_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
	if (patternfound)
	{
		cornerSubPix(gray_img, corners, Size(sub_radius, sub_radius), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cvtColor(gray_img, out, CV_GRAY2BGR);
		drawChessboardCorners(out, pattern_size, corners, true);
	}
	return patternfound;
}

bool CornerFinder::FindCornersOfOmniCamera(const cv::Mat &img, std::vector<cv::Point2f> &corners, const cv::Size& pattern_size, int sub_radius /*= 11*/, cv::Mat &out /*= cv::Mat()*/)
{
	cv::Mat gray_img;
	if (img.channels() == 3)
		cvtColor(img, gray_img, CV_BGR2GRAY);
	else if (img.channels() == 1)
		img.copyTo(gray_img);
	else
		return false;

	corners.clear();
	int patternfound =FindCornersForOmniCamera(img, pattern_size, corners);
	if (patternfound==1)
	{
		cornerSubPix(img, corners, Size(sub_radius, sub_radius), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		AdjustCornerDirection(corners, pattern_size);
		cvtColor(img, out, CV_GRAY2BGR);
		drawChessboardCorners(out, pattern_size, corners, true);
	}
	return patternfound==1;
}

bool CornerFinder::ReadCornersFromMatlab(const std::string &path,std::vector<std::vector<cv::Point2f>> &corners,cv::Size &pattern_size)
{
// 	cv::FileStorage fs2("E:\\Data\\omni+Velodyne\\ScreenShots0517\\Zebra\\Corners.yaml", cv::FileStorage::APPEND);

	cv::FileStorage fs(path, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		return false;
	}
	cv::Mat board_size_mat, board_count_mat;
	fs["board_size"] >> board_size_mat;
	fs["board_count"] >> board_count_mat;
	size_t board_count = board_count_mat.at<double>(0, 0);
	pattern_size = cv::Size(board_size_mat.at<double>(0, 1), board_size_mat.at<double>(0, 0));

	corners.resize(board_count);

	for (size_t i = 0; i < board_count; i++)
	{
		stringstream ss;
		ss << "corners" << i + 1;
		string temp;
		ss >> temp;
		cv::Mat pts;
		fs[temp] >> pts;

		if (pts.empty())
		{
			corners[i] = (std::vector<cv::Point2f>());
		}
		else
		{
			std::vector<cv::Point2f> temp_corners;
			for (size_t i = 0; i < pattern_size.width*pattern_size.height; i++)
				temp_corners.push_back(cv::Point2f(pts.at<double>(i, 0), pts.at<double>(i, 1)));
			corners[i] = temp_corners;
		}
	}

	return true;
}



