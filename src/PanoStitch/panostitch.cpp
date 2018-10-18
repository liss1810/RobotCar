#include "panostitch.h"
#include "Common_GQ.h"
#include <iostream>

#include <AprilTags/TagDetector.h>
#include <AprilTags/TagDetection.h>
#include <AprilTags/Tag36h11.h>

using namespace std;
using namespace cv;
using namespace detail;
using namespace Common_GQ;

PanoStitch::PanoStitch()
{
	pano_w = 8000;
	pano_h = 4000;
	//全景图大小
	panoSize.width = pano_w;
	panoSize.height = pano_h;
	//全景图
	Mat pano(panoSize, CV_8UC3, Scalar::all(0));
}

PanoStitch::~PanoStitch()
{
}

//计算掩码
//************************************
// Method:    ComputeImageMask---生成扭曲图掩码
// FullName:  PanoStitch::ComputeImageMask
// Access:    public 
// Returns:   bool
// Qualifier:
// Parameter: const cv::Mat & image
// Parameter: cv::Mat & imageMask
// Parameter: int dilateSize
// Parameter: int erodeSize
//************************************
bool PanoStitch::ComputeImageMask(const cv::Mat &image, cv::Mat &imageMask, int dilateSize /*= -1*/, int erodeSize /*= -1*/)
{
	imageMask.create(image.rows, image.cols, CV_8UC1);
	if (image.channels() == 4)
	{
		cv::Mat rgbImage(image.size(), CV_8UC3);
		for (int i = 0; i < image.rows; ++i)
		{
			uchar* rowData = (uchar*)(image.data) + image.cols*i * 4;
			uchar* rowRGBData = (uchar*)(rgbImage.data) + rgbImage.cols*i * 3;
			uchar* rowMask = (uchar*)(imageMask.data) + imageMask.cols*i;
			for (int j = 0; j < image.cols; ++j)
			{
				*rowRGBData++ = *rowData++;
				*rowRGBData++ = *rowData++;
				*rowRGBData++ = *rowData++;
				*rowMask++ = *rowData++;;
			}
		}
	}
	else if (image.channels() == 3)
	{
		for (int i = 0; i < image.rows; ++i)
		{
			uchar* rowData = (uchar*)(image.data) + image.cols*i * 3;
			uchar* rowMask = (uchar*)(imageMask.data) + imageMask.cols*i;
			for (int j = 0; j < image.cols; ++j)
			{
				if (rowData[0] || rowData[1] || rowData[2])
					*rowMask++ = 255;
				else
					*rowMask++ = 0;
				//*rowMask++ = ( rowData[0]!=0 || rowData[1]!=0 || rowData[2]!=0 )*255;
				rowData += 3;
			}
		}
	}
	else
		return false;
	if (dilateSize > 0)
	{
		cv::Mat kernel1(dilateSize, dilateSize, CV_8UC1, cv::Scalar(1));
		cv::dilate(imageMask, imageMask, kernel1);
	}

	if (erodeSize > 0)
	{
		cv::Mat kernel2(erodeSize, erodeSize, CV_8UC1, cv::Scalar(1));
		cv::erode(imageMask, imageMask, kernel2);
	}
	return true;
}
//加载相机参数
bool PanoStitch::LoadCameraParam(Mat& K, Mat& D, Mat& R, Mat& T, Mat &xi, char location, std::string filepath_rt, std::string filepath_kd) {

	FileStorage fs(filepath_rt, FileStorage::READ);
	FileStorage fs_kd(filepath_kd, FileStorage::READ);
	if (!fs.isOpened() && !fs_kd.isOpened()) {
		cout << "open file error!" << endl;
		return false;
	}

	switch (location)
	{
	case 'l':
		fs_kd["Kl"] >> K;
		fs_kd["Dl"] >> D;
		fs_kd["Xil"] >> xi;
		fs["Rl"] >> R;
		fs["Tl"] >> T;
		break;
	case 'c':
		fs_kd["Kc"] >> K;
		fs_kd["Dc"] >> D;
		fs_kd["Xic"] >> xi;
		fs["Rc"] >> R;
		fs["Tc"] >> T;
		break;
	case 'r':
		fs_kd["Kr"] >> K;
		fs_kd["Dr"] >> D;
		fs_kd["Xir"] >> xi;
		fs["Rr"] >> R;
		fs["Tr"] >> T;
		break;
	default:
		cout << "input char error!";
		break;
	}
	fs.release();
	fs_kd.release();
	return true;
}

//************************************
// Method:    OpenCVSpherical----使用OpenCV的球面投影
// FullName:  PanoStitch::OpenCVSpherical
// Access:    public 
// Returns:   bool
// Qualifier:
// Parameter: const cv::Mat K
// Parameter: const cv::Mat D
// Parameter: const std::vector<cv::Mat> Rs
// Parameter: const std::string paths
// Parameter: int num_pic
// Parameter: cv::Size panoSize
//************************************
bool PanoStitch::OpenCVSpherical(const cv::Mat K, const cv::Mat D, const std::vector<cv::Mat> Rs,const std::string paths,int num_pic,cv::Size panoSize,CAM_MODEL model) {

	vector<string> path_imgs;
	path_imgs = Common_GQ::GetListFiles(paths, "jpg");
	vector<Mat> imgs(num_pic);
	vector<Mat> warped_images(num_pic);
	for (size_t i = 0; i < num_pic; i++)
	{
		imgs[i] = imread(path_imgs[i]);

		Mat K_, mapx, mapy, img_undis;
		Mat warped_image;
		Mat pic = imgs[i];

		Point pt;
		Mat pano_rgb(panoSize, CV_8UC3, 0.0);

		if (model==CAM_FISHEYE)
		{
			fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, pic.size(), Mat::eye(3, 3, CV_64FC1), K_, 1.0, panoSize);
			fisheye::initUndistortRectifyMap(K, D, Mat::eye(3, 3, CV_64FC1), K_, pic.size(), CV_32FC1, mapx, mapy);
			remap(pic, img_undis, mapx, mapy, CV_INTER_AREA);
		}
		else
		{
			K.copyTo(K_);
			pic.copyTo(img_undis);
		}

		detail::SphericalWarper test_warp(panoSize.width / 2.0 / M_PI);
		K_.convertTo(K_, CV_32FC1);
		pt = test_warp.warp(img_undis, K_, Rs[i], CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
		cout << pt << endl;

		Rect roi = Rect(pt.x + panoSize.width / 2, pt.y , warped_image.cols, warped_image.rows);
		if (roi.x < 0)
			roi.x = 0;
		if (roi.br().x >= panoSize.width)
			roi.width = panoSize.width - roi.x;
		if (roi.y < 0)
			roi.y = 0;
		if (roi.br().y >= panoSize.height)
			roi.height = panoSize.height - roi.y;
		warped_image.copyTo(pano_rgb(roi));
		pano_rgb.copyTo(warped_images[i]);
		cout << endl;

		stringstream ss;
		string str;
		ss << i << ".png";
		ss >> str;
		cv::imwrite(str, warped_images[i]);
		system(str.c_str());
	}

	return true;
}

//************************************
// Method:    OmniSpherical----高强大哥的全向相机模型
// FullName:  PanoStitch::OmniSphericalKO
// Access:    public 
// Returns:   bool
// Qualifier:
// Parameter: const cv::Mat m_K
// Parameter: const cv::Mat m_D
// Parameter: const std::vector<cv::Mat> Rs
// Parameter: const std::string paths
// Parameter: int num_pic
// Parameter: cv::Size panoSize
// Parameter: double fov
// Parameter: double radian_offset
//************************************
bool PanoStitch::OmniSpherical(const cv::Mat m_K, const cv::Mat m_D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize, double fov, double radian_offset) {
	
	//全景图的宽
	int pano_width = panoSize.width;
	//全景图的高
	int pano_height = panoSize.height;
	//全景图映射表
	Mat pano_mapx, pano_mapy;
	//全景图
	Mat pano;
	//X方向--每像素弧度
	const double phi_each = 2 * M_PI / pano_width;
	//Z方向--每像素弧度
	const double theta_each = M_PI / pano_height;
	pano_mapx = cv::Mat(pano_height, pano_width, CV_32FC1, -1.0);
	pano_mapy = cv::Mat(pano_height, pano_width, CV_32FC1, -1.0);
	float *mapx_data = (float*)pano_mapx.data, *mapy_data = (float*)pano_mapy.data;
	//使用Eigen进行矩阵计算的效率高
	Eigen::Matrix3d K_eigen = Common_GQ::toMatrix3d(m_K);
	//提取K的值
	Vec2d f(m_K.at<double>(0, 0), m_K.at<double>(1, 1));
	Vec2d c(m_K.at<double>(0, 2), m_K.at<double>(1, 2));
	Vec2d k(m_D.at<double>(0), m_D.at<double>(1));
	Vec2d p(m_D.at<double>(2), m_D.at<double>(3));
	double s = m_K.at<double>(0, 1);
	//xi定为单位距离
	double xi = 1;
	//畸变参数
	double p1 = p[0], p2 = p[1];
	double k1 = k[0], k2 = k[1];
	// 视场角转化为弧度值
	const double thre = M_PI / 180 * fov;
	//转换坐标轴
	Eigen::Matrix3d ref_cen;
	ref_cen << 0, -1, 0, 0, 0, -1, -1, 0, 0;

	vector<Mat> warped_imgs(num_pic);

	for (int a = 0; a < num_pic; a++)
	{
		//外参R
		Eigen::Matrix3d R_Eigen = Common_GQ::toMatrix3d(Rs[a]);
		//调整坐标轴方向
		R_Eigen = R_Eigen*ref_cen;

		for (int i = 0; i < pano_height; i++)
		{
			for (int j = 0; j < pano_width; j++)
			{
				// phi与X轴夹角（0,2PI）
				double phi = j * phi_each + radian_offset;
				// theta与Z轴夹角(0,PI)
				double theta = i * theta_each;

				Eigen::Vector3d Xs_(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
				Xs_ = R_Eigen*Xs_;
				Vec3d Xs(Xs_(0), Xs_(1), Xs_(2));

				// convert to normalized image plane
				Vec2d xu = Vec2d(Xs[0] / (Xs[2] + xi), Xs[1] / (Xs[2] + xi));

				// add distortion
				Vec2d xd;
				double r2 = xu[0] * xu[0] + xu[1] * xu[1];
				double r4 = r2 * r2;
				xd[0] = xu[0] * (1 + k1 * r2 + k2 * r4) + 2 * p1*xu[0] * xu[1] + p2 * (r2 + 2 * xu[0] * xu[0]);
				xd[1] = xu[1] * (1 + k1 * r2 + k2 * r4) + p1 * (r2 + 2 * xu[1] * xu[1]) + 2 * p2*xu[0] * xu[1];

				// convert to pixel coordinate
				Vec2d px;
				px[0] = f[0] * xd[0] + s * xd[1] + c[0];
				px[1] = f[1] * xd[1] + c[1];

				if (theta < thre)
				{
					mapx_data[j + i * pano_width] = px[0];
					mapy_data[j + i * pano_width] = px[1];
				}
			}
		}

		Mat img;
		vector<string> path_imgs(num_pic);
		path_imgs = Common_GQ::GetListFiles(paths, "jpg");
		img = imread(path_imgs[a]);
		cv::remap(img, img, pano_mapx, pano_mapy, CV_INTER_LINEAR);
		warped_imgs[a] = img;
		stringstream ss;
		ss << a << ".jpg";
		string s;
		ss >> s;
		imwrite(s, img);
		system(s.c_str());
	}

	//去中间45°范围内的像素
	double wid = (M_PI / 2)*(1.0 / phi_each);
	//像素宽度
	cout << warped_imgs.size() << endl << wid << endl;
	
	return true;
}


bool PanoStitch::FishEyeModelSpherical(const cv::Mat m_K, const cv::Mat m_D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize, double fov, double radian_offset) {

	//全景图的宽
	int pano_width = panoSize.width;
	//全景图的高
	int pano_height = panoSize.height;
	//全景图映射表
	Mat pano_mapx, pano_mapy;
	//全景图
	Mat pano;
	//X方向--每像素弧度
	const double phi_each = 2 * M_PI / pano_width;
	//Z方向--每像素弧度
	const double theta_each = M_PI / pano_height;
	pano_mapx = cv::Mat(pano_height, pano_width, CV_32FC1, -1.0);
	pano_mapy = cv::Mat(pano_height, pano_width, CV_32FC1, -1.0);
	float *mapx_data = (float*)pano_mapx.data, *mapy_data = (float*)pano_mapy.data;
	//使用Eigen进行矩阵计算的效率高
	Eigen::Matrix3d K_eigen = Common_GQ::toMatrix3d(m_K);
	//提取K的值
	Vec2d f(m_K.at<double>(0, 0), m_K.at<double>(1, 1));
	Vec2d c(m_K.at<double>(0, 2), m_K.at<double>(1, 2));
	Vec2d k(m_D.at<double>(0), m_D.at<double>(1));
	Vec2d p(m_D.at<double>(2), m_D.at<double>(3));
	double s = m_K.at<double>(0, 1);
	//xi定为单位距离
	double xi = 1;
	//畸变参数
	double p1 = p[0], p2 = p[1];
	double k1 = k[0], k2 = k[1];
	// 视场角转化为弧度值
	const double thre = M_PI / 180 * fov;
	//转换坐标轴
	Eigen::Matrix3d ref_cen;
	ref_cen << 0, -1, 0, 0, 0, -1, -1, 0, 0;

	vector<Mat> warped_imgs(num_pic);

	for (int a = 0; a < num_pic; a++)
	{
		//外参R
		Eigen::Matrix3d R_Eigen = Common_GQ::toMatrix3d(Rs[a]);
		//调整坐标轴方向
		R_Eigen = R_Eigen*ref_cen;

		for (int i = 0; i < pano_height; i++)
		{
			for (int j = 0; j < pano_width; j++)
			{
// 				// phi与X轴夹角（0,2PI）
// 				double phi = j * phi_each + radian_offset;
// 				// theta与Z轴夹角(0,PI)
// 				double theta = i * theta_each;
// 
// 				Eigen::Vector3d Xs_(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
// 				Xs_ = R_Eigen*Xs_;
// 				Vec3d Xs(Xs_(0), Xs_(1), Xs_(2));
// 
// 				// convert to normalized image plane
// 				Vec2d xu = Vec2d(Xs[0] / (Xs[2]+xi), Xs[1] / (Xs[2] + xi));
// 
// 				// add distortion
// 				double r2 = xu[0] * xu[0] + xu[1] * xu[1];
// 				double r = sqrt(r2);
// 				double sita = atan(r);
// 				double xd = sita*(1 + k1*pow(sita, 2) + k2*pow(sita, 4) + p1*pow(sita, 6), p2*pow(sita, 8));
// 
// 				//convert to the distorted point coordinates
// 				Vec2d px_d;
// 				px_d[0] = (xd / r)*xu[0];
// 				px_d[1] = (xd / r)*xu[1];
// 
// 				// convert to pixel coordinate
// 				Vec2d px;
// 				px[0] = f[0] * (px_d[0] + s * px_d[1]) + c[0];
// 				px[1] = f[1] * px_d[1] + c[1];
// 
// 				if (theta < thre)
// 				{
// 					mapx_data[j + i * pano_width] = px[0];
// 					mapy_data[j + i * pano_width] = px[1];
// 				}

				// phi与X轴夹角（0,2PI）
				double phi = j * phi_each + radian_offset;
				// theta与Z轴夹角(0,PI)
				double theta = i * theta_each;

				Eigen::Vector3d Xs_(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
				Xs_ = R_Eigen*Xs_;
				Vec3d Xs(Xs_(0), Xs_(1), Xs_(2));

				// convert to normalized image plane
				Vec2d xu = Vec2d(Xs[0] / (Xs[2] + xi), Xs[1] / (Xs[2] + xi));

				// add distortion
				Vec2d xd;
				double r2 = xu[0] * xu[0] + xu[1] * xu[1];
				double r4 = r2 * r2;
				xd[0] = xu[0] * (1 + k1 * r2 + k2 * r4) + 2 * p1*xu[0] * xu[1] + p2 * (r2 + 2 * xu[0] * xu[0]);
				xd[1] = xu[1] * (1 + k1 * r2 + k2 * r4) + p1 * (r2 + 2 * xu[1] * xu[1]) + 2 * p2*xu[0] * xu[1];

				// convert to pixel coordinate
				Vec2d px;
				px[0] = f[0] * xd[0] + s * xd[1] + c[0];
				px[1] = f[1] * xd[1] + c[1];

				if (theta < thre)
				{
					mapx_data[j + i * pano_width] = px[0];
					mapy_data[j + i * pano_width] = px[1];
				}

			}
		}

		Mat img;
		vector<string> path_imgs(num_pic);
		path_imgs = Common_GQ::GetListFiles(paths, "jpg");
		img = imread(path_imgs[a]);
		cv::remap(img, img, pano_mapx, pano_mapy, CV_INTER_LINEAR);
		warped_imgs[a] = img;
		stringstream ss;
		ss << a << ".jpg";
		string s;
		ss >> s;
		imwrite(s, img);
		system(s.c_str());
	}



	return true;
}

//Omnidirectional--全向相机模型
bool PanoStitch::OmniDModelPanoMap(cv::Mat &mapx, cv::Mat &mapy, Mat m_K, Mat m_D, Mat m_R, Mat m_T, Mat m_xi, int pano_width, int pano_height, double fov, double radian_offset) {

	double r = pano_width / M_PI / 2;
	//
	const double phi_each = -2 * M_PI / pano_width;
	//
	const double theta_each = M_PI / pano_height;

	mapx = cv::Mat(pano_height, pano_width, CV_32FC1, -1.0);
	mapy = cv::Mat(pano_height, pano_width, CV_32FC1, -1.0);

	float *mapx_data = (float*)mapx.data, *mapy_data = (float*)mapy.data;

	Eigen::Matrix3d K_eigen = Common_GQ::toMatrix3d(m_K);

	Vec2d f(m_K.at<double>(0, 0), m_K.at<double>(1, 1));
	Vec2d c(m_K.at<double>(0, 2), m_K.at<double>(1, 2));
	Vec2d k(m_D.at<double>(0), m_D.at<double>(1));
	Vec2d p(m_D.at<double>(2), m_D.at<double>(3));
	double s = m_K.at<double>(0, 1);
	double xi = m_xi.at<double>(0);
	double p1 = p[0], p2 = p[1];
	double k1 = k[0], k2 = k[1];
	// 视场角转化为弧度值
	const double thre = M_PI / 180 * fov;

	Mat R_c, T_c, R_l, T_l;
	Mat xi_c, xi_l;
	LoadCameraParam(Mat(3, 3, CV_64FC1), Mat(4, 1, CV_64FC1), R_c, T_c, xi_c, 'c');
	LoadCameraParam(Mat(3, 3, CV_64FC1), Mat(4, 1, CV_64FC1), R_l, T_l, xi_l, 'l');

// 	Mat Rtc, Rtl, RTT;
// 
// 	Rtc = Common_GQ::toMat44(R_c, T_c).inv();
// 	Rtl = Common_GQ::toMat44(R_l, T_l).inv();
// 	RTT = Rtc.inv()*Rtl*Rtl;
// 
// 	Mat R, T;
// 	Common_GQ::toMat33AndMat31(RTT, R, T);
// 	cout << "RX" << R << endl;
	// 	Eigen::Matrix3d R_Eigen = Common_GQ::toMatrix3d(R_l*R_c.inv());
	Eigen::Matrix3d R_Eigen = Common_GQ::toMatrix3d(m_R);

	Eigen::Matrix3d ref_cen;
	ref_cen << 0, 1, 0, 1, 0, 0, 0, 0, -1;
	R_Eigen = R_Eigen*ref_cen;
#pragma omp parallel for
	for (int i = 0; i < pano_height; i++)
	{
		for (int j = 0; j < pano_width; j++)
		{
			// phi与X轴夹角（0,2PI）
			double phi = j * phi_each + radian_offset;
			// theta与Z轴夹角(0,PI)
			double theta = i * theta_each;


			Eigen::Vector3d Xs_(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
			Xs_ = R_Eigen*Xs_;

			Vec3d Xs(Xs_(0), Xs_(1), Xs_(2));
			// convert to center camera coordinate
			//cout << R << endl << "Xs:" << Xs << endl;
			// convert to normalized image plane
			Vec2d xu = Vec2d(Xs[0] / (Xs[2] + xi), Xs[1] / (Xs[2] + xi));

			// add distortion
			Vec2d xd;
			double r2 = xu[0] * xu[0] + xu[1] * xu[1];
			double r4 = r2 * r2;
			xd[0] = xu[0] * (1 + k1 * r2 + k2 * r4) + 2 * p1*xu[0] * xu[1] + p2 * (r2 + 2 * xu[0] * xu[0]);
			xd[1] = xu[1] * (1 + k1 * r2 + k2 * r4) + p1 * (r2 + 2 * xu[1] * xu[1]) + 2 * p2*xu[0] * xu[1];

			// convert to pixel coordinate
			Vec2d px;
			px[0] = f[0] * xd[0] + s * xd[1] + c[0];
			px[1] = f[1] * xd[1] + c[1];

			// 			if (theta < thre)
			{
				mapx_data[j + i * pano_width] = px[0];
				mapy_data[j + i * pano_width] = px[1];
			}
		}
	}
	return true;
}

void PanoStitch::OmniModel() {

	//全景图映射表
	Mat pano_mapx, pano_mapy;
	//全景图的长
	int pano_width = 2048;
	//相机参数
	Mat K, D, R, T, xi;
	//加载参数
	LoadCameraParam(K, D, R, T, xi, 'c');
	cout << K << D << R << T << xi << endl;
	bool r = OmniDModelPanoMap(pano_mapx, pano_mapy, K, D, Mat::eye(3, 3, CV_64FC1), T, xi, pano_width, pano_width / 2);
	if (!r)
	{
		cout << "Generate camera pano map failed!" << endl;
		return;
	}

	Mat img;
	img = imread("D:\\Data\\RobotCar\\src\\1403537938670876_c.png", -1);
	cvtColor(img, img, CV_BayerRG2RGB);

	cv::remap(img, img, pano_mapx, pano_mapy, CV_INTER_LINEAR);
	stringstream ss;
	ss << "dst.png";
	imwrite(ss.str(), img);
	system("dst.png");
}

//鱼眼----会使图片失真
void PanoStitch::FishEyeModel(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths) {

	//获取彩色图
	vector<Mat> imgs(paths.size());
	for (size_t i = 0; i < paths.size(); i++)
	{
		Mat img_test1 = imread(paths[i], -1);
		Mat flag;
		cvtColor(img_test1, imgs[i], CV_BayerRG2RGB);


		//cvtColor(img_test1, flag, CV_BayerRG2RGB);
		//resize(flag, imgs[i],Size(4000,4000));
	}

	vector<Mat> RTs(NUM_IMG), RTT(NUM_IMG), RT_dst(NUM_IMG);

	for (size_t i = 0; i < NUM_IMG; i++)
	{
		Rs[i].convertTo(Rs[i], CV_32FC1);
		Ts[i].convertTo(Ts[i], CV_32FC1);
		//将坐标系转化到参考相机坐标系
		RTs[i] = Common_GQ::toMat44(Rs[i], Ts[i]);
		std::cout << "RTS:" << RTs[i] << endl;
	}
	Mat R, T, img_test;

	//参照相机---中间相机
	float data[] = { 4.0000000000000000e+02, 0., 5.0844171921102975e+02,
		0.,4.0000084665689131e+02, 4.9731390857987373e+02,
		0., 0., 1. };
	Mat KK1(3, 3, CV_32FC1, data);

	cout << KK1 << endl;
	vector<Mat> warped_images(NUM_IMG);
	for (size_t i = 0; i < NUM_IMG; i++)
	{

		if (i != 1)
		{
			RTT[i] = RTs[1].inv()*RTs[i] * RTs[i];
		}
		else
		{
			RTT[i] = Mat::eye(4, 4, CV_32FC1);
		}
		img_test = imgs[i];
		Common_GQ::toMat33AndMat31(RTs[i], R, T);
		std::cout << "RR:" << R << endl;

		cv::Size img_size(img_test.cols, img_test.rows);
		Mat warped_image;
		Point point;
		Mat K_new, mapx, mapy, img_undis;

		fisheye::estimateNewCameraMatrixForUndistortRectify(Ks[i], Ds[i], img_size, Mat::eye(3, 3, CV_64FC1), K_new, 1.0, img_size);
		fisheye::initUndistortRectifyMap(Ks[i], Ds[i], Mat::eye(3, 3, CV_64FC1), K_new, img_size, CV_32FC1, mapx, mapy);
		cv::remap(img_test, img_undis, mapx, mapy, CV_INTER_AREA);

		detail::SphericalWarper test_warp(panoSize.width / (M_PI * 2.0));
		K_new.convertTo(K_new, CV_32FC1);
		R.convertTo(R, CV_32FC1);
		Point pt = test_warp.warp(img_undis, K_new, R, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
		warped_images[i] = warped_image;
		cout << i << '\t' << pt << endl;
		cout << endl;
		stringstream ss;
		string str;
		ss << i << ".png";
		ss >> str;
		cv::imwrite(str, warped_images[i]);
		system(str.c_str());


	}

#if 0
	vector<Mat> warped_images(NUM_IMG);

	vector<Mat> RTs(NUM_IMG), RTT(NUM_IMG), RT_dst(NUM_IMG);

	for (size_t i = 0; i < NUM_IMG; i++)
	{
		Rs[i].convertTo(Rs[i], CV_32FC1);
		Ts[i].convertTo(Ts[i], CV_32FC1);
		RTs[i] = Common_GQ::toMat44(Rs[i], Ts[i]).inv();
		std::cout << "RTS:" << RTs[i] << endl;

	}
	Mat R, T;

	for (size_t i = 0; i < NUM_IMG; i++)
	{

		if (i != 1)
		{
			RTT[i] = RTs[1].inv()*RTs[i] * RTs[i];
		}
		else
		{
			RTT[i] = Mat::eye(4, 4, CV_32FC1);
		}
		Common_GQ::toMat33AndMat31(RTT[i], R, T);
		std::cout << "RR:" << R << endl;

		Mat img_test = imread(paths[i], -1);
		cvtColor(img_test, img_test, CV_BayerRG2RGB);

		stringstream s;
		string str1;
		s << i << "origin.png";
		s >> str1;
		cv::imwrite(str1, img_test);

		cv::Size img_size(img_test.cols, img_test.rows);
		Mat warped_image;
		Mat pano_rgb(panoSize, CV_8UC3, 0.0);
		Point point;

		Mat K_new, mapx, mapy, img_undis, pano_img;

		fisheye::estimateNewCameraMatrixForUndistortRectify(Ks[i], Ds[i], img_size, Mat::eye(3, 3, CV_64FC1), K_new, 1.0, img_size);
		fisheye::initUndistortRectifyMap(Ks[i], Ds[i], Mat::eye(3, 3, CV_64FC1), K_new, img_size, CV_32FC1, mapx, mapy);
		cv::remap(img_test, img_undis, mapx, mapy, CV_INTER_CUBIC);

		detail::SphericalWarper test_warp(panoSize.width / 2 / 2.0 / M_PI);
		K_new.convertTo(K_new, CV_32FC1);

		point = test_warp.warp(img_undis, K_new, R, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);

		//point.x =0;
		//point.y =0;
		std::cout << i << '\t' << point << endl;
		std::cout << T << endl;


		Rect roi = Rect(point.x + pano_w / 2, -point.y + pano_h / 2, warped_image.cols, warped_image.rows);
		if (roi.x < 0)
			roi.x = 0;
		if (roi.br().x >= pano_w)
			roi.width = pano_w - roi.x;
		if (roi.y < 0)
			roi.y = 0;
		if (roi.br().y >= pano_h)
			roi.height = pano_h - roi.y;
		warped_image.copyTo(pano_rgb(roi));
		pano_rgb.copyTo(warped_images[i]);


		stringstream ss;
		string str;
		ss << i << ".png";
		ss >> str;
		cv::imwrite(str, warped_images[i]);
		system(str.c_str());


	}
#endif



}

//普通
void PanoStitch::OriginModel(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths) {

	vector<Mat> warped_images(NUM_IMG);
	vector<Mat> RTs(NUM_IMG), RTT(NUM_IMG), RT_dst(NUM_IMG);
	vector<Point> points(NUM_IMG);
	Mat R, T, K1;
	for (size_t i = 0; i < 3; i++)
	{
		Rs[i].convertTo(Rs[i], CV_32FC1);
		Ts[i].convertTo(Ts[i], CV_32FC1);
		RTs[i] = Common_GQ::toMat44(Rs[i], Ts[i]).inv();

		cout << "RTS:" << RTs[i] << endl;
	}

	Mat pano_rgb(panoSize, CV_8UC3, 0.0);
	for (size_t i = 0; i < NUM_IMG; i++)
	{

		if (i != 1)
		{
			RTT[i] = RTs[1].inv()*RTs[i] * RTs[i];
		}
		else
		{
			RTT[i] = Mat::eye(4, 4, CV_32FC1);
		}


		Common_GQ::toMat33AndMat31(RTT[i], R, T);
		cout << "\nRR:" << R << endl << "TT" << T << endl;

		Mat warped_image;
		Point point;
		Mat K_new, mapx, mapy, img_undis, pano_img;
		detail::SphericalWarper test_warp(panoSize.width / M_PI / 2.0);

		Mat image = imread(paths[i]);
		//Ks[1].convertTo(K1, CV_32FC1);
		Ks[i].convertTo(K1, CV_32FC1);
		R.convertTo(R, CV_32FC1);
		point = test_warp.warp(image, K1, R, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
		points[i] = point;
		
		cout << i << '\t' << point << endl;

		Rect roi = Rect(point.x + pano_w / 2, point.y, warped_image.cols, warped_image.rows);
		if (roi.x < 0)
			roi.x = 0;
		if (roi.br().x >= pano_w)
			roi.width = pano_w - roi.x;
		if (roi.y < 0)
			roi.y = 0;
		if (roi.br().y >= pano_h)
			roi.height = pano_h - roi.y;
		warped_image.copyTo(pano_rgb(roi));
		pano_rgb.copyTo(warped_images[i]);

		stringstream ss;
		string str;
		ss << i << ".png";
		ss >> str;
		cv::imwrite(str, warped_images[i]);
		system(str.c_str());
	}

}

//PTGui
//************************************
// Method:    PTGuiModel----利用ptGUI脚本进行拼接，作为备选方案
// FullName:  PanoStitch::PTGuiModel
// Access:    public 
// Returns:   bool
// Qualifier:
// Parameter: const char * ptgui_script_path
// Parameter: const std::string img_paths
//************************************
bool PanoStitch::PTGuiModel(const char* ptgui_script_path, const std::string img_paths)  {

	vector<string> paths = Common_GQ::GetListFiles(img_paths, "jpg");

	int images_count = paths.size();
	int camera_type;
	cv::Rect m_rect;
	std::vector<cv::Mat> m_mapsx, m_mapsy;

	fullPath *ImageFileNames = new fullPath[images_count];
	fullPath scriptFileName;
	float **map_ptr = new float*[images_count];
	int width, height;
	PTRect rect;
	for (int i = 0; i < images_count; i++)
		strcpy((ImageFileNames + i)->name, paths.at(i).c_str());
	strcpy(scriptFileName.name, ptgui_script_path);
	int result = panoCreatePanoramaToLookupTable(ImageFileNames, images_count, &scriptFileName, &map_ptr, &width, &height, &rect);
	if (result == 0)
	{
		m_rect = cv::Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
		panoSize = cv::Size(width, height);
		m_mapsx.clear();
		m_mapsy.clear();
		m_mapsx.resize(images_count);
		m_mapsy.resize(images_count);
#pragma omp parallel for
		for (int k = 0; k < images_count; k++)
		{
			cv::Mat mapx(height, width, CV_32FC1);
			cv::Mat mapy(height, width, CV_32FC1);
			float *recx_data = (float*)(mapx.data);
			float *recy_data = (float*)(mapy.data);
			for (size_t i = 0; i < height; i++)
			{
				for (size_t j = 0; j < width; j++)
				{
					recx_data[i*width + j] = map_ptr[k][(i*width + j) * 2];
					recy_data[i*width + j] = map_ptr[k][(i*width + j) * 2 + 1];
				}
			}
			mapx.copyTo(m_mapsx[k]);
			mapy.copyTo(m_mapsy[k]);
		}
		panoLookupTableRelease(&map_ptr, images_count);
	}
	delete ImageFileNames;
	delete map_ptr;
	vector<Mat> distort_imgs(images_count);
	vector<Mat> warped_imgs(images_count);
	vector<Mat> warped_imgs_mask(images_count);
	for (int i=0;i<images_count;i++)
	{
		Mat tmp = imread(paths[i]);
		tmp.copyTo(distort_imgs[i]);
		remap( distort_imgs[i], warped_imgs[i], m_mapsx[i], m_mapsy[i], CV_INTER_AREA);

		//掩码
		Mat tmp_mask;
		ComputeImageMask(warped_imgs[i], tmp_mask);
		warped_imgs_mask[i] = tmp_mask;
	}
	
	return true;
}

//帽子加权算法
bool PanoStitch::LinearBlending(CAM_TYPE type, std::vector<cv::Mat> &warped_imgs, const std::vector<cv::Mat> &masks, cv::Mat &blended_img, int index,Size m_pano_size)
{
	//全景图大小
	int row = m_pano_size.height, col = m_pano_size.width;
	Mat result(row, col, CV_8UC3, Scalar(0, 0, 0));

	//重合区域查找表
	Mat weight_c(row, col, CV_8UC2, Scalar(0, 0));
#pragma omp parallel for
	for (int a = 0; a < masks.size(); a++)
	{
		Mat flag_a, flag_a1, flag_a2;
		int a2, a3;
		a2 = (a + 1) >= masks.size() ? (a + 1 - masks.size()) : (a + 1);
		a3 = (a + 2) >= masks.size() ? (a + 2 - masks.size()) : (a + 2);

		flag_a = masks[a];
		flag_a1 = masks[a2];
		flag_a2 = masks[a3];

		for (int b = 0; b < flag_a.rows; b++)
		{
			uchar* row_a = flag_a.ptr(b);
			uchar* row_a1 = flag_a1.ptr(b);
			uchar* row_a2 = flag_a2.ptr(b);
			uchar* row_w = weight_c.ptr(b);
			for (int c = 0; c < flag_a.cols; c++) {
				if ((row_a[c] & row_a1[c] & (!row_a2[c]))) {
					row_w[c * 2] = a2;
					row_w[c * 2 + 1] = a;

				}
			}
		}
	}
#pragma omp parallel for
	for (int i = 0; i < weight_c.rows; i++) {

		vector<int> weigs;
		int num1, num2;
		int center = 0;

		//计算重合区域长度  
		uchar* row_w = weight_c.ptr(i);
		for (int j = 0; j < weight_c.cols; j++) {
			if (j < weight_c.cols - 1) {
				num1 = (int)row_w[j * 2];
				num2 = (int)row_w[j * 2 + 1];
				if ((num1 == (int)(row_w[(j + 1) * 2])) && (num2 == (int)(row_w[(j + 1) * 2 + 1]))) {
					center++;
				}
				else {
					weigs.emplace_back(center);
					center = 0;
				}
			}
			else {
				weigs.emplace_back(center);
			}
		}

		int w = 0;
		row_w = weight_c.ptr(i);
		uchar* row_r = result.ptr(i);
		int sum_center = 0;
		//计算比例,以中心点为界
		for (int m = 0; m < weight_c.cols; m++) {

			num1 = (int)row_w[m * 2];
			num2 = (int)row_w[m * 2 + 1];

			if ((m < weight_c.cols - 1)) {

				if (num1 == (int)(row_w[(m + 1) * 2]) && num2 == (int)(row_w[(m + 1) * 2 + 1])) {
					//不是边界，按距离比融合	
					row_r[m * 3] = ((warped_imgs[num1].at<Vec3b>(i, m)[0] * (1.0 - (double)(m - sum_center) / weigs[w])) + (warped_imgs[num2].at<Vec3b>(i, m)[0] * ((double)(m - sum_center) / weigs[w])));
					row_r[m * 3 + 1] = ((warped_imgs[num1].at<Vec3b>(i, m)[1] * (1.0 - (double)(m - sum_center) / weigs[w])) + (warped_imgs[num2].at<Vec3b>(i, m)[1] * ((double)(m - sum_center) / weigs[w])));
					row_r[m * 3 + 2] = ((warped_imgs[num1].at<Vec3b>(i, m)[2] * (1.0 - (double)(m - sum_center) / weigs[w])) + (warped_imgs[num2].at<Vec3b>(i, m)[2] * ((double)(m - sum_center) / weigs[w])));
				}
				else {
					//边界处直接取第二张图片像素值	
					row_r[m * 3] = warped_imgs[num2].at<Vec3b>(i, m)[0];
					row_r[m * 3 + 1] = warped_imgs[num2].at<Vec3b>(i, m)[1];
					row_r[m * 3 + 2] = warped_imgs[num2].at<Vec3b>(i, m)[2];

					sum_center += weigs[w];
					w++;
				}
			}
			else {
				//最后一列
				row_r[m * 3] = (warped_imgs[num2].at<Vec3b>(i, m)[0]);
				row_r[m * 3 + 1] = (warped_imgs[num2].at<Vec3b>(i, m)[1]);
				row_r[m * 3 + 2] = (warped_imgs[num2].at<Vec3b>(i, m)[2]);
			}
		}
	}
	blended_img = result;
	return true;
}

//统一到同一坐标系
bool PanoStitch::FishModel_Test(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths) {

	//获取彩色图
	vector<Mat> imgs(paths.size());
	vector<Mat> undis_imgs(paths.size());
	vector<Mat> warped_images(paths.size());
	for (size_t i = 0; i < paths.size(); i++)
	{
		Mat img_test1 = imread(paths[i], -1);
		cvtColor(img_test1, imgs[i], CV_BayerRG2RGB);

		cv::Size img_size(1024, 1024);
		Mat K_new;
		Mat mapx, mapy;

		//鱼眼相机模型，减少视场角损失
		fisheye::estimateNewCameraMatrixForUndistortRectify(Ks[i], Ds[i], img_size, Mat::eye(3, 3, CV_64FC1), K_new, 1.0, img_size);
		//去畸变
		fisheye::initUndistortRectifyMap(Ks[i], Ds[i], Mat::eye(3, 3, CV_64FC1), K_new, img_size, CV_32FC1, mapx, mapy);
		cv::remap(imgs[i], undis_imgs[i], mapx, mapy, CV_INTER_AREA);

		//球面投影
		detail::SphericalWarper test_warp(3072 / (M_PI * 2.0));
		Mat R_ = Mat::eye(3, 3, CV_32FC1);
		K_new.convertTo(K_new, CV_32FC1);
		test_warp.warp(undis_imgs[i], K_new, R_, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_images[i]);
	}

	//虚拟相机的内参
	Eigen::Matrix3d KK;
	KK << 400, 0, 600, 0, 400, 600, 0, 0, 1;
	//cout << KK;
	Mat dst(3072, 3072, CV_8UC3, Scalar::all(0));

	Mat flag = warped_images[1];
	Size img_size = flag.size();
	//cout << img_size << endl;
	Eigen::Matrix3d K = Common_GQ::toMatrix3d(Ks[1]);
	R = Rs[1], T = Ts[1];
	//convert to camera coor
	Eigen::Matrix4d RT_ = Common_GQ::toMatrix4d(Common_GQ::toMat44(R, T));
	//
	Eigen::Vector4d point_c(0, 0, 0, 1);
	Mat mapx(3072, 3072, CV_32FC1), mapy(3072, 3072, CV_32FC1);
	for (int i = 0; i < 3072; i++)
	{
		for (int j = 0; j < 3072; j++)
		{
			//像素点
			Eigen::Vector3d point_p(i, j, 1);
			//相机坐标系
			Eigen::Vector3d p = K.inverse()*point_p;
			//cout << K << endl << p << endl;
			point_c[0] = p[0], point_c[1] = p[1], point_c[2] = p[2];
			//cout << point_c << endl;
			Eigen::Vector4d pp = RT_.inverse()*point_c;
			//cout << pp << endl;
			Eigen::Vector3d pp_(pp[0], pp[1], pp[2]);
			Eigen::Vector3d m = KK*pp_;
			//cout << m << endl;
			mapx.at<float>(i, j) = m[0]+1000;
			mapy.at<float>(i, j) = m[1]+500;
		}
	}
	remap(flag, dst, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT);
	imwrite("dst.jpg", dst);
	system("dst.jpg");

// 	for (int n = 0; n < imgs.size(); n++)
// 	{
// 		cv::Mat tmp = warped_images[n];
// 		//相机坐标系三维点
// 		Eigen::Vector4d point_c(0, 0, 0, 1);
// 		//世界坐标系三维点
// 		Eigen::Vector4d point_w;
// 
// 		Eigen::Matrix3d K = Common_GQ::toMatrix3d(Ks[n]);
// 		cv::Mat R, T, RR;
// 		R = Rs[n], T = Ts[n];
// 		//R = Mat::eye(3, 3, CV_64FC1), T = Ts[n];
// 		RR = Mat::eye(4, 4, CV_64FC1);
// 
// 		//坐标转换
// 		double data_r[] = { 0,-1,0,0,0,-1,1,0,0 };
// 		cv::Mat ref_r(3, 3, CV_64FC1, data_r);
// 		R *= ref_r;
// 
// 		//参考系到相机
// 		Eigen::Matrix4d RT_ = Common_GQ::toMatrix4d(Common_GQ::toMat44(R, T));
// 		//单位矩阵
// 		Eigen::Matrix4d RT = Common_GQ::toMatrix4d(RR);
// 
// 		Mat mapx(3072, 3072, CV_32FC1), mapy(3072, 3072, CV_32FC1);
// 
// 		for (int i = 0; i < 3027; i++)
// 		{
// 			for (int j = 0; j < 3027; j++)
// 			{
// 				//像素坐标点
// 				Eigen::Vector3d point_p(i, j, 1);
// 				//转到相机坐标系
// 				Eigen::Vector3d p = K.inverse()*point_p;
// 				point_c[0] = p[0], point_c[1] = p[1], point_c[2] = p[2];
// 				
// 				//虚拟相机坐标系
// 				Eigen::Vector4d pp = RT_*point_c;
// 				//虚拟相机坐标系转到像素坐标系
// 				Eigen::Vector3d pp_(pp[0], pp[1], pp[2]);
// 				Eigen::Vector3d m = KK*pp_;
// 
// 				mapx.at<float>(i, j) = m[0];
// 				mapy.at<float>(i, j) = m[1];	
// 			}
// 		}
// 		
// 		remap(tmp, dst, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT);
// 		stringstream ss;
// 		string str;
// 		ss << n << ".png";
// 		ss >> str;
// 		imwrite(str, dst);
// 		system(str.c_str());
// 		cout << endl;
// 	}
	return true;
}

//
bool PanoStitch::ApriltagModel(std::vector<cv::Mat> src_images, std::vector<cameraPara> paras,float markLength) {

	cv::Mat gray_img,tmp;
	tmp = src_images[0];
	tmp.copyTo(gray_img);
	cv::cvtColor(gray_img, gray_img, COLOR_BGR2GRAY);
	AprilTags::TagDetector atg(AprilTags::tagCodes36h11);
	vector<AprilTags::TagDetection> detections = atg.extractTags(gray_img);
	Eigen::Matrix4d tr;
	Eigen::Matrix3d H;
	std::vector<cv::Point3d> points;

	if (detections.size()>0)
	{
		for (size_t i = 0; i < detections.size(); i++)
		{
			points = detections[i].getObjectPoints(markLength);
			H = detections[i].homography;
			cout << "H:" << H << endl;
			for (int j = 0; j < points.size(); j++)
				cout << "\n角点坐标：" << endl << points[j] << endl;
			// draw results
			detections[i].draw(tmp);
			cv::Mat K = paras[0].K;
			//世界坐标系转化为相机坐标
			tr = detections[i].getRelativeTransform(markLength, K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2));
			cout << "tag坐标系到相机坐标系RT:\n" << tr << endl;
		}
	}

	Mat dst; 
	cout << Common_GQ::toCvMat(H) << endl;
	warpPerspective(tmp, dst, Common_GQ::toCvMat(H), panoSize);



	return true;

}

//************************************
// Method:    KinectTest
// FullName:  PanoStitch::KinectTest
// Access:    public 
// Returns:   bool
// Qualifier:
// Parameter: const std::vector<cv::Mat> K
// Parameter: const cv::Mat D
// Parameter: const std::vector<cv::Mat> Rs
// Parameter: const std::string paths
// Parameter: int num_pic
// Parameter: cv::Size panoSize
//************************************
bool PanoStitch::KinectTest(const std::vector<cv::Mat> K, const cv::Mat D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize) {


	vector<string> path_imgs;
	path_imgs = Common_GQ::GetListFiles(paths, "png");
	vector<Mat> imgs(num_pic);
	vector<Mat> warped_images(num_pic);
	for (size_t i = 0; i < num_pic; i++)
	{
		imgs[i] = imread(path_imgs[i]);

		Mat K_, mapx, mapy, img_undis,R_;
		Mat warped_image;
		Mat pic = imgs[i];

		Point pt;
		Mat pano_rgb(panoSize, CV_8UC3, 0.0);

		K_ = K[i];
		R_ = Rs[i];
		detail::SphericalWarper test_warp(panoSize.width / 2.0 / M_PI);
		K_.convertTo(K_, CV_32FC1);
		R_.convertTo(R_, CV_32FC1);
		pt = test_warp.warp(pic, K_, R_, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
		cout << pt << endl;

		Rect roi = Rect(pt.x + panoSize.width / 2, pt.y, warped_image.cols, warped_image.rows);
		if (roi.x < 0)
			roi.x = 0;
		if (roi.br().x >= panoSize.width)
			roi.width = panoSize.width - roi.x;
		if (roi.y < 0)
			roi.y = 0;
		if (roi.br().y >= panoSize.height)
			roi.height = panoSize.height - roi.y;
		warped_image.copyTo(pano_rgb(roi));
		pano_rgb.copyTo(warped_images[i]);
		cout << endl;

		stringstream ss;
		string str;
		ss << i << ".png";
		ss >> str;
		cv::imwrite(str, warped_images[i]);
		system(str.c_str());
	}

	return true;

}


bool PanoStitch::BirdEyeKinectTest(const cv::Mat K, const cv::Mat D, std::vector<cv::Point2f> corners, cv::Mat& src, cv::Mat&dst, int num_pic, int board_w, int board_h) {

	cv::Mat gray_image;
	cvtColor(src, gray_image, CV_BGR2GRAY);
	
	//
	cv::Size board_sz(board_w,board_h);
	if (corners.size() <= 0)
	{
		cout << "corners size is null" << endl;
		return false;
	}

	cv::Point2f objPts[4], imgPts[4];
	objPts[0].x = 0;
	objPts[0].y = 0;
	objPts[1].x = (board_w - 1);
	objPts[1].y = 0;
	objPts[2].x = 0;
	objPts[2].y = (board_h - 1);
	objPts[3].x = (board_w - 1);
	objPts[3].y = (board_h - 1);

	imgPts[0] = corners[0];
	imgPts[1] = corners[board_h*(board_w - 1)]; 
	imgPts[2] = corners[board_h-1];
	imgPts[3] = corners[board_h*board_w - 1];

// 	cv::circle(gray_image, imgPts[0], 9, cv::Scalar(255, 0, 0), 3);
// 	cv::circle(gray_image, imgPts[1], 9, cv::Scalar(0, 255, 0), 3);
// 	cv::circle(gray_image, imgPts[2], 9, cv::Scalar(0, 0, 255), 3);
// 	cv::circle(gray_image, imgPts[3], 9, cv::Scalar(0, 255, 255), 3);
// 	
// 	cv::drawChessboardCorners(gray_image, board_sz, corners, true);
	
	cv::Mat H = cv::getPerspectiveTransform(objPts, imgPts);
	cout << H << endl;
	double Z = 35;
	H.at<double>(2, 2) = 35;
	cv::Size sz(2000,4000);
	cv::warpPerspective(src,dst, H,src.size(),   cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar::all(0));
	imwrite("dst.png", dst);
	system("dst.png");
	return true;
}

//************************************
// Method:    BirdEyePano---车载环视鸟瞰图
// FullName:  PanoStitch::BirdEyePano
// Access:    public 
// Returesrns:   bool
// Qualifier:
//************************************
bool PanoStitch::BirdEyePano(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths) {

	int num = paths.size();
	vector<Mat> imgs(num);
	vector<Mat> warped_images(num);
	vector<Mat> undis_imgs(num);
	cv::Size img_size;

	for (size_t i = 0; i < num; i++)
	{
		Mat img_test = imread(paths[i], -1);
		cvtColor(img_test, imgs[i], CV_BayerRG2RGB);
		Mat K_new;
		Mat mapx, mapy;
		img_size = imgs[i].size();
		//鱼眼相机模型，减少视场角损失
		fisheye::estimateNewCameraMatrixForUndistortRectify(Ks[i], Ds[i], img_size, Mat::eye(3, 3, CV_64FC1), K_new, 1.0, img_size);
		//去畸变
		fisheye::initUndistortRectifyMap(Ks[i], Ds[i], Mat::eye(3, 3, CV_64FC1), K_new, img_size, CV_32FC1, mapx, mapy);
		cv::remap(imgs[i], undis_imgs[i], mapx, mapy, CV_INTER_AREA);

		Mat R = Rs[i];
		Mat K = Ks[i];
		float data[] = {1,0,0,0,1,0,0,0,1};
		Mat ref(3, 3, CV_32FC1, data);
		cout << ref << endl;
		R = ref*R;
		Mat warped_image;
		detail::PlaneWarper wp(1.f);
		R.convertTo(R, CV_32FC1);
		K.convertTo(K, CV_32FC1);
		wp.warp(undis_imgs[i], K, R, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
		warped_images[i] = warped_image;
	}

	return true;
}