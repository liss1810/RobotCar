#include "Common_GQ.h"
#include <QDir>
#include <QFileInfoList>
#include <Eigen/Geometry>
#include <QDebug>
using namespace std;

int win7CompareStr(const char *sa, const char *sb) 
{
	const char *psa = sa;
	const char *psb = sb;
	for (;;)
	{
		if (*psa != '\0' && *psb != '\0')
		{
			if (*psa >= '0' && *psa <= '9' && *psb >= '0' && *psb <= '9')
			{
				const char *psa2 = psa;
				const char *psb2 = psb;
				int ia, ib;
				char abuf[20];
				char bbuf[20];
				int na, nb;

				ia = 0;
				while (*psa2 != '\0' && *psa2 >= '0' && *psa2 <= '9')
				{
					abuf[ia++] = *psa2;
					++psa2;
				}
				abuf[ia] = '\0';
				na = atoi(abuf);

				ib = 0;
				while (*psb2 != '\0' && *psb2 >= '0' && *psb2 <= '9')
				{
					bbuf[ib++] = *psb2;
					++psb2;
				}
				bbuf[ib] = '\0';
				nb = atoi(bbuf);

				if (na == nb)
				{
					psa += ia;
					psb += ia;
				}
				else if (na < nb)
				{
					return -1;
				}
				else
				{
					return 1;
				}
			}
			else
			{
				if (*psa < *psb)
				{
					return -1;
				}
				else if (*psa == *psb)
				{
					++psa;
					++psb;
				}
				else
				{
					return 1;
				}
			}
		}
		else if (*psa == '\0' && *psb != '\0')
		{
			return -1;
		}
		else if (*psa == '\0' && *psb == '\0')
		{
			return 0;
		}
		else
		{
			return 1;
		}
	}
}
bool StringCompare(const string& a, const string& b)
{
	int result = win7CompareStr(a.c_str(), b.c_str());
	if (result == -1)
		return true;
	else
		return false;
}
int64 Common_GQ::getRunTimeStart()
{
	return cv::getTickCount();
}

int64 Common_GQ::getRunTimeEnd(int64 start, std::string string)
{
	float time = (cv::getTickCount() - start) / cv::getTickFrequency();
	std::cout << string << " Running Time : " << time << " seconds." << std::endl;
	return cv::getTickCount();
}

vector<string> Common_GQ::GetListFolders(const string &path)
{
	vector<string> list;
	QDir dir(QString::fromLocal8Bit(path.c_str()));
	QDir::Filters filter = QDir::NoDotAndDotDot | QDir::Dirs;
	QFileInfoList info_list = dir.entryInfoList(filter);
	for (int i = 0; i < info_list.size(); i++)
	{
		list.push_back(info_list[i].absoluteFilePath().toLocal8Bit().constData());
	}
	std::sort(list.begin(), list.end(), StringCompare);
	return list;
}
vector<string> Common_GQ::GetListFiles(const string &path, const string &suffix)
{
	vector<string> list;
	QDir dir(QString::fromLocal8Bit(path.c_str()));
	QDir::Filters filter = QDir::NoDotAndDotDot | QDir::Files;
	QFileInfoList info_list = dir.entryInfoList(filter);
	QString suffix_str(suffix.c_str());
	for (int i = 0; i < info_list.size(); i++)
		if (suffix.empty() || info_list[i].suffix() == suffix_str)
			list.push_back(info_list[i].absoluteFilePath().toLocal8Bit().constData());
	std::sort(list.begin(), list.end(), StringCompare);
	return list;
}

std::vector<cv::Mat> Common_GQ::toDescriptorVector(const cv::Mat &Descriptors)
{
	std::vector<cv::Mat> vDesc;
	vDesc.reserve(Descriptors.rows);
	for (int j = 0;j < Descriptors.rows;j++)
		vDesc.push_back(Descriptors.row(j));

	return vDesc;
}

cv::Mat Common_GQ::toCvMat(const Eigen::Matrix<double, 4, 4> &m)
{
	cv::Mat cvMat(4, 4, CV_64F);
	for (int i = 0;i < 4;i++)
		for (int j = 0; j < 4; j++)
			cvMat.at<double>(i, j) = m(i, j);

	return cvMat.clone();
}

cv::Mat Common_GQ::toMat44(const cv::Mat &R, const cv::Mat &T)
{
	assert(R.type() == T.type());
	cv::Mat new_R;
	if (R.cols == 1)
	{
		cv::Rodrigues(R, new_R);
	}
	else
		R.copyTo(new_R);
	cv::Mat cvMat = cv::Mat::eye(4, 4, CV_64F);
	if (R.type()==CV_64F)
	{
		for (int i = 0;i < 3;i++)
		{
			for (int j = 0;j < 3;j++)
			{
				cvMat.at<double>(i, j) = new_R.at<double>(i, j);
			}
		}
		for (int i = 0;i < 3;i++)
		{
			cvMat.at<double>(i, 3) = T.at<double>(i, 0);
		}
	}
	else if (R.type() == CV_32F)
	{
		for (int i = 0;i < 3;i++)
		{
			for (int j = 0;j < 3;j++)
			{
				cvMat.at<double>(i, j) = new_R.at<float>(i, j);
			}
		}
		for (int i = 0;i < 3;i++)
		{
			cvMat.at<double>(i, 3) = T.at<float>(i, 0);
		}
	}
	return cvMat.clone();
}

bool Common_GQ::toMat33AndMat31(const cv::Mat &RT, cv::Mat &R, cv::Mat &T)
{
	if (RT.cols!=4||RT.rows!=4)
	{
		return false;
	}
	cv::Rect r(0, 0, 3, 3);
	RT(r).copyTo(R);
	r.x = 3;
	r.width = 1;
	RT(r).copyTo(T);
	return true;
}

cv::Mat Common_GQ::toCvMat(const Eigen::Matrix<double, 3, 3>& R, const Eigen::Matrix<double, 3, 1>& t)
{
	cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
	for (int i = 0;i < 3;i++)
	{
		for (int j = 0;j < 3;j++)
		{
			cvMat.at<float>(i, j) = R(i, j);
		}
	}
	for (int i = 0;i < 3;i++)
	{
		cvMat.at<float>(i, 3) = t(i);
	}

	return cvMat.clone();
}

Eigen::Matrix<double, 3, 1> Common_GQ::toVector3d(const cv::Mat & cvVector)
{
	Eigen::Matrix<double, 3, 1> v;
	if (cvVector.type()==CV_32FC1)
		v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);
	else if (cvVector.type()==CV_64FC1)
		v << cvVector.at<double>(0), cvVector.at<double>(1), cvVector.at<double>(2);

	return v;
}

Eigen::Matrix<double, 3, 1> Common_GQ::toVector3d(const cv::Point3f & cvPoint)
{
	Eigen::Matrix<double, 3, 1> v;
	v << cvPoint.x, cvPoint.y, cvPoint.z;

	return v;
}

Eigen::Matrix<double, 3, 3> Common_GQ::toMatrix3d(const cv::Mat & cvMat3)
{
	Eigen::Matrix<double, 3, 3> M;
	if (cvMat3.type()==CV_32FC1)
	{
		M << cvMat3.at<float>(0, 0), cvMat3.at<float>(0, 1), cvMat3.at<float>(0, 2),
			cvMat3.at<float>(1, 0), cvMat3.at<float>(1, 1), cvMat3.at<float>(1, 2),
			cvMat3.at<float>(2, 0), cvMat3.at<float>(2, 1), cvMat3.at<float>(2, 2);
	}
	else if (cvMat3.type() == CV_64FC1)
	{
		M << cvMat3.at<double>(0, 0), cvMat3.at<double>(0, 1), cvMat3.at<double>(0, 2),
			cvMat3.at<double>(1, 0), cvMat3.at<double>(1, 1), cvMat3.at<double>(1, 2),
			cvMat3.at<double>(2, 0), cvMat3.at<double>(2, 1), cvMat3.at<double>(2, 2);
	}
	return M;
}
Eigen::Matrix<double, 4, 4> Common_GQ::toMatrix4d(const cv::Mat & cvMat44)
{
	Eigen::Matrix<double, 4, 4> M;

	if (cvMat44.type()==CV_32FC1)
	{
		M << cvMat44.at<float>(0, 0), cvMat44.at<float>(0, 1), cvMat44.at<float>(0, 2), cvMat44.at<float>(0, 3),
			cvMat44.at<float>(1, 0), cvMat44.at<float>(1, 1), cvMat44.at<float>(1, 2), cvMat44.at<float>(1, 3),
			cvMat44.at<float>(2, 0), cvMat44.at<float>(2, 1), cvMat44.at<float>(2, 2), cvMat44.at<float>(2, 3),
			cvMat44.at<float>(3, 0), cvMat44.at<float>(3, 1), cvMat44.at<float>(3, 2), cvMat44.at<float>(3, 3);
	} 
	else if(cvMat44.type()==CV_64FC1)
	{
		M << cvMat44.at<double>(0, 0), cvMat44.at<double>(0, 1), cvMat44.at<double>(0, 2), cvMat44.at<double>(0, 3),
			cvMat44.at<double>(1, 0), cvMat44.at<double>(1, 1), cvMat44.at<double>(1, 2), cvMat44.at<double>(1, 3),
			cvMat44.at<double>(2, 0), cvMat44.at<double>(2, 1), cvMat44.at<double>(2, 2), cvMat44.at<double>(2, 3),
			cvMat44.at<double>(3, 0), cvMat44.at<double>(3, 1), cvMat44.at<double>(3, 2), cvMat44.at<double>(3, 3);
	}
	return M;
}

std::vector<double> Common_GQ::toQuaternion(const cv::Mat & M)
{
	Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
	Eigen::Quaterniond q(eigMat);
	std::vector<double> v(4);
	v[0] = q.x();
	v[1] = q.y();
	v[2] = q.z();
	v[3] = q.w();

	return v;
}

cv::Mat Common_GQ::toCvMat(const Eigen::Matrix3d &m)
{
	cv::Mat cvMat(3, 3, CV_64F);
	for (int i = 0;i < 3;i++)
		for (int j = 0; j < 3; j++)
			cvMat.at<double>(i, j) = m(i, j);

	return cvMat.clone();
}

cv::Mat Common_GQ::toCvMat(const Eigen::Matrix<double, 3, 1> &m)
{
	cv::Mat cvMat(3, 1, CV_64F);
	for (int i = 0; i < 3; i++)
		cvMat.at<double>(i) = m(i);
	return cvMat.clone();
}

void Common_GQ::MatDeleteOneCol(cv::Mat &mat, int col)
{
	int cols = mat.cols, rows = mat.rows;
	cv::Mat temp(rows, cols - 1, CV_64FC1);
	if (col < 0 || col >= cols)
	{
		return;
	}
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols - 1; j++)
		{
			if (j < col)
			{
				temp.at<double>(i, j) = mat.at<double>(i, j);
			}
			else
			{
				temp.at<double>(i, j) = mat.at<double>(i, j + 1);
			}
		}
	}
	temp.copyTo(mat);
}
void Common_GQ::MatDeleteOneRow(cv::Mat &mat, int row)
{
	int cols = mat.cols, rows = mat.rows;
	cv::Mat temp(rows - 1, cols, CV_64FC1);
	if (row < 0 || row >= rows)
	{
		return;
	}
	for (size_t i = 0; i < rows - 1; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			if (i < row)
			{
				temp.at<double>(i, j) = mat.at<double>(i, j);
			}
			else
			{
				temp.at<double>(i, j) = mat.at<double>(i + 1, j);
			}
		}
	}
	temp.copyTo(mat);
}
bool Common_GQ::MatElementBitAnd(const cv::Mat& mat1, const cv::Mat &mat2, cv::Mat &out)
{
	if (mat1.type() != CV_64FC1 || mat2.type() != CV_64FC1 || mat1.cols != mat2.cols || mat1.rows != mat2.rows)
	{
		return false;
	}
	int width = mat1.cols;
	int height = mat1.rows;
	out.create(height, width, CV_64FC1);
	for (int i = 0;i < height;i++)
		for (int j = 0;j < width;j++)
			out.at<double>(i, j) = ((mat1.at<double>(i, j) == 0 || mat2.at<double>(i, j) == 0) ? 0 : 1);
	return true;
}
bool Common_GQ::MatElementZero(const cv::Mat& mat, cv::Mat &out)
{
	if (mat.type() != CV_64FC1)
	{
		return false;
	}
	out.create(mat.rows, mat.cols, mat.type());

	int width = mat.cols;
	int height = mat.rows;
	for (int i = 0;i < height;i++)
	{
		for (int j = 0;j < width;j++)
			if (mat.at<double>(i, j) == 0.0)
				out.at<double>(i, j) = 1.;
			else
				out.at<double>(i, j) = 0.;

	}
	return true;
}

QImage Common_GQ::cvMat2QImage(const cv::Mat& mat)
{
	// 8-bits unsigned, NO. OF CHANNELS = 1  
	if (mat.type() == CV_8UC1)
	{
		QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
		// Set the color table (used to translate colour indexes to qRgb values)  
		image.setColorCount(256);
		for (int i = 0; i < 256; i++)
		{
			image.setColor(i, qRgb(i, i, i));
		}
		// Copy input Mat  
		uchar *pSrc = mat.data;
		for (int row = 0; row < mat.rows; row++)
		{
			uchar *pDest = image.scanLine(row);
			memcpy(pDest, pSrc, mat.cols);
			pSrc += mat.step;
		}
		return image;
	}
	// 8-bits unsigned, NO. OF CHANNELS = 3  
	else if (mat.type() == CV_8UC3)
	{
		// Copy input Mat  
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat  
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	else if (mat.type() == CV_8UC4)
	{
		qDebug() << "CV_8UC4";
		// Copy input Mat  
		const uchar *pSrc = (const uchar*)mat.data;
		// Create QImage with same dimensions as input Mat  
		QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
		return image.copy();
	}
	else
	{
		qDebug() << "ERROR: Mat could not be converted to QImage.";
		return QImage();
	}
}

cv::Mat Common_GQ::QImage2cvMat(const QImage &image)
{
	cv::Mat mat;
	qDebug() << image.format();
	switch (image.format())
	{
	case QImage::Format_ARGB32:
	case QImage::Format_RGB32:
	case QImage::Format_ARGB32_Premultiplied:
		mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
		break;
	case QImage::Format_RGB888:
		mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
		cv::cvtColor(mat, mat, CV_BGR2RGB);
		break;
	case QImage::Format_Indexed8:
		mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
		break;
	}
	return mat;
}