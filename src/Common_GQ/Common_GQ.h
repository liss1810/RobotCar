#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <QImage>

namespace Common_GQ
{
	/************************************************************************/
	/* 计时函数                                                                     */
	/************************************************************************/
	int64 getRunTimeStart();
	int64 getRunTimeEnd(int64 start, std::string string);

	/************************************************************************/
	/* 文件操作                                                                     */
	/************************************************************************/
	std::vector<std::string> GetListFolders(const std::string &path);
	std::vector<std::string> GetListFiles(const std::string &path, const std::string &suffix = "");

	/************************************************************************/
	/* 矩阵转换                                                                     */
	/************************************************************************/
	std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);
	cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
	cv::Mat toCvMat(const Eigen::Matrix3d &m);
	cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
	cv::Mat toCvMat(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);
	cv::Mat toMat44(const cv::Mat &R, const cv::Mat &T);
	bool toMat33AndMat31(const cv::Mat &RT, cv::Mat &R, cv::Mat &T);
	Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);
	Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);
	Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);
	Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat & cvMat44);
	std::vector<double> toQuaternion(const cv::Mat &M);

	void MatDeleteOneCol(cv::Mat &mat, int col);

	void MatDeleteOneRow(cv::Mat &mat, int row);

	bool MatElementZero(const cv::Mat& mat, cv::Mat &out);

	QImage cvMat2QImage(const cv::Mat& mat);

	cv::Mat QImage2cvMat(const QImage &image);

	bool MatElementBitAnd(const cv::Mat& mat1, const cv::Mat &mat2, cv::Mat &out);


};

