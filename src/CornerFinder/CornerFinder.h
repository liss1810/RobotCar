#pragma once
#include <opencv2/opencv.hpp>

#define MULT_THREAD 1

struct C_Pyramid
{

	cv::Mat m_MIm;			// downsize image
	cv::Mat m_MImadj;		// adaptively adjust
	cv::Mat m_MIme;			// sobel edge image
	cv::Mat m_MIx;
	cv::Mat m_MIy;
	int m_scale;

	std::vector<cv::Point2d> ctcrnrs;
	std::vector<cv::Point2d> ctcrnrpts;
	std::vector<std::vector<int>> ctpeaklocs;
	int ctnocrnrs;

};
class CornerFinder
{

public:
	CornerFinder();
	~CornerFinder();


	static void CalcChessboardCorners(cv::Size boardSize, float squareSize, int nums, std::vector<std::vector<cv::Point3f>> & object_Points);

	static bool FindCornersOfOpenCV(const cv::Mat &img, std::vector<cv::Point2f> &corners, const cv::Size& pattern_size, int sub_radius, cv::Mat &out /*= cv::Mat()*/);

	bool FindCornersOfOmniCamera(const cv::Mat &img, std::vector<cv::Point2f> &corners, const cv::Size& pattern_size, int sub_radius /*= 11*/, cv::Mat &out /*= cv::Mat()*/);

	static bool ReadCornersFromMatlab(const std::string &path, std::vector<std::vector<cv::Point2f>> &corners, cv::Size &pattern_size);

	std::vector<cv::Point2f> FindSeparateCorners(const cv::Mat &img, cv::Mat &out = cv::Mat());

	std::vector<cv::Point2f> FindSubPixCorners(const cv::Mat &img, const std::vector<cv::Point2f>& pts, int win_radius = 5) const;

	int FindCornersOfAutoFixMissingCorners(const cv::Mat &img, std::vector<cv::Point2f> &corners, cv::Mat &out = cv::Mat());


public:


	int             m_iNoLevels;	// no of Pyramid levels
	C_Pyramid*		m_pPyr;			// Pyramid
	int				m_iHWin;		// Harris window size
	double			m_dTh;			// Parameter to adjust adaptive thresholding

private:
	int FindCornersForOmniCamera(const cv::Mat& img, const cv::Size &pattern_size, std::vector<cv::Point2f> &chessbord_corners);
	int AdjustCornerDirection(std::vector<cv::Point2f> &pts, const cv::Size &pattern_size, int origin_pos = 0);

};
