#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "Common_GQ/Common_GQ.h"
#include <math.h>
#define _USE_MATH_DEFINES
#define PI 3.1415926
#define NUM_IMG 3

using namespace std;
using namespace cv;
using namespace detail;
using namespace Common_GQ;


int pano_w = 8000;
int pano_h = 4000;
//全景图大小
cv::Size panoSize(pano_w, pano_h);
//全景图
Mat pano(panoSize, CV_8UC3, Scalar::all(0));

bool rectify() {
	cv::Mat imag = imread("D:/Data/RobotCar/src/1403537878679423_r.png");

	Mat K = cv::Mat::eye(3, 3, CV_64FC1), K1;

	//相机内参
	K.at<double>(0, 0) = 4.0000000000000000e+002;
	K.at<double>(1, 1) = 4.0000000000000000e+002;
	K.at<double>(0, 2) = 5.0250375400000000e+002;
	K.at<double>(1, 2) = 4.9025903300000000e+002;

	// 	/*圆周鱼眼，畸变参数 k1,k2,p1,p2 */
	// 	double data[] = { -1.4075127613371857e-001, -8.9736194824215788e-003,
	// 						4.8634042031003143e-003, 1.4425043608206586e-005}; 	//畸变参数
	Mat  D(4, 1, CV_64FC1, Scalar::all(0));
	// 
	// 	double dataR[] = { -7.06890166e-001, 1.71275977e-002, -7.07115889e-001,
	// 						1.15949260e-002, 9.99853075e-001, 1.26269842e-002,
	// 						7.07228303e-001, 7.26935163e-004, -7.06984878e-001 };
	// 	//旋转矩阵
	// 	Mat R(3, 3, CV_64FC1, dataR);

	FileStorage fs("D:/Data/RobotCar/camera_calibration/intrinsic/R.yaml", FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "open file error" << endl;
		return false;
	}


	//fs["Kr"] >> K;
	fs["Dr"] >> D;


	cout << K << endl << D << endl;

	Mat res;
	Size imageSize = imag.size();

	cv::Mat map1, map2;
	//initUndistortRectifyMap(K, D, Mat(), Mat(),imageSize, CV_16SC2, map1, map2);
	//cv::initUndistortRectifyMap(K, D, Mat(), K, imageSize, CV_16SC2, map1, map2);
	//cv::remap(imag, res, map1, map2, cv::INTER_LINEAR);

	//K.copyTo(K1);
	//调节视场大小,乘的系数越小视场越大
	//K1.at<double>(0, 0) *= 0.8;
	//K1.at<double>(1, 1) *= 0.8;
	//调节校正图中心，建议置于校正图中心
	//K1.at<double>(0, 2) = 0.5 * imag.cols;
	//K1.at<double>(1, 2) = 0.5 * imag.rows;

	//fisheye::undistortImage(imag, res, K, D, K1,imageSize);
	cv::Mat P;
	cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, imageSize, cv::Mat::eye(3, 3, CV_32FC1), P, 1.0);
	fisheye::initUndistortRectifyMap(P, D, Mat(), K, imageSize, CV_32FC1, map1, map2);
	cv::remap(imag, res, map1, map2, cv::INTER_LINEAR);

	imwrite("test.jpg", res);
	system("test.jpg");

	return true;
}

bool writeEtrinsic() {
	string filePath = "D:/Data/RobotCar/camera_calibration/extrinsic/mono_right.yaml";
	FileStorage fs(filePath, FileStorage::WRITE);
	double data_r[] = {
		0.2896   , 0.9503   , 0.1144 ,
		-0.9254   , 0.2475   , 0.2870 ,
		0.2444 ,-0.1889   , 0.9511 };

	double data_t[] = { -0.2587,-1.6810, 0.3226 };

	cv::Mat R(3, 3, CV_64FC1, data_r);
	cv::Mat T(3, 1, CV_64FC1, data_t);

	cout << R << endl << T << endl;

	fs << "R" << R;
	fs << "T" << T;
	fs.release();
	return true;
}

bool initPara(Mat& K, Mat& D, Mat& R, Mat& T,char location,string filepath_rt,string filepath_kd) {

	FileStorage fs(filepath_rt, FileStorage::READ);
	FileStorage fs_kd(filepath_kd, FileStorage::READ);
	if (!fs.isOpened()&&!fs_kd.isOpened()) {
		cout << "open file error!" << endl;
		return false;
	}

	switch (location)
	{
	case 'l':
		fs_kd["Kl"] >> K;
		fs_kd["Dl"] >> D;
		fs["Rl"] >> R;
		fs["Tl"] >> T;
		break;
	case 'c':
		fs_kd["Kc"] >> K;
 		fs_kd["Dc"] >> D;
		fs["Rc"] >> R;
		fs["Tc"] >> T;
		break;
	case 'r':
		fs_kd["Kr"] >> K;
		fs_kd["Dr"] >> D;
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


bool LoadCameraParam(Mat& K, Mat& D, Mat& R, Mat& T, Mat &xi ,char location, string filepath_rt="D:/Data/RobotCar/camera_calibration/extrinsic/RT.yaml", string filepath_kd="D:/Data/RobotCar/camera_calibration/OmniCamera/KD.yaml") {

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
//获取两张图片的特征点匹配数
int findMatchPoints(Mat src_1, Mat src_2) {

	Ptr<FeaturesFinder> finder;
	//采用ORB算法寻找特征点
	finder = new OrbFeaturesFinder();
	vector<ImageFeatures> features(2);

	Mat grayP1, grayP2;
	vector<Mat> images_gray;
	//灰度化
	cvtColor(src_1, grayP1, CV_BGR2GRAY);
	images_gray.emplace_back(grayP1);
	cvtColor(src_2, grayP2, CV_BGR2GRAY);
	images_gray.emplace_back(grayP2);

	//寻找特征点
	for (int i = 0; i < images_gray.size(); i++)
	{
		Mat img = images_gray[i];
		(*finder)(img, features[i]);
		features[i].img_idx = i;
		cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << endl;
	}
	//释放内存
	finder->collectGarbage();
	//特征点匹配
	vector<MatchesInfo> pair_matches;
	BestOf2NearestMatcher matcher(false);
	matcher(features, pair_matches);
	matcher.collectGarbage();
	cout << "匹配数：" << pair_matches[1].num_inliers << endl<<"size:"<<endl;
	
	return pair_matches[1].num_inliers;
}

bool ComputeImageMask(const cv::Mat &image, cv::Mat &imageMask, int dilateSize = -1 /*= -1*/, int erodeSize = -1 /*= -1*/)
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

//球面投影
void GenerateUndistortedImgToSphereProjectionMap(cv::Mat & mapx, cv::Mat & mapy, const cv::Mat & K, const cv::Mat & RT, int pano_width, int pano_height)
{
	

	double r = pano_width / M_PI / 2;
	const double phi_each = -2 * M_PI / pano_width;
	const double theta_each = M_PI / pano_height;

	mapx = cv::Mat(pano_height, pano_width, CV_32FC1, 0.0);
	mapy = cv::Mat(pano_height, pano_width, CV_32FC1, 0.0);
	float *mapx_data = (float*)mapx.data, *mapy_data = (float*)mapy.data;

	Eigen::Matrix4d RT_eigen = toMatrix4d(RT);            //小矩阵操作用Eigen比Opencv快得多
	Eigen::Matrix3d K_eigen = toMatrix3d(K);

#pragma omp parallel for
	for (int i = 0; i < pano_height; i++)
	{
		for (int j = 0; j < pano_width; j++)
		{
			double phi = j*phi_each + M_PI;
			double theta = i*theta_each;
			Eigen::Vector4d xyz1(r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta), 1.0);
			xyz1 = RT_eigen*xyz1;
			if (xyz1(2) > 0)            //只取该镜头看得见的部分
			{
				Eigen::Vector3d xyz(xyz1(0), xyz1(1), xyz1(2));
				xyz = K_eigen*xyz;
				mapx_data[j + i*pano_width] = xyz(0) / xyz(2);
				mapy_data[j + i*pano_width] = xyz(1) / xyz(2);
			}
		}
	}
	
}

//鱼眼----会使图片失真
void FishEyeModel(const vector<Mat>& Ks, const vector<Mat>& Ds, vector<Mat>& Ts, vector<Mat>& Rs, vector<string>& paths) {

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
	Mat KK1(3, 3, CV_32FC1,data);

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
		Point pt=test_warp.warp(img_undis, K_new, R, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
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

		detail::SphericalWarper test_warp(panoSize.width/2 / 2.0 / M_PI);
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
void OriginModel(const vector <Mat> &Ks, const vector<Mat>& Ds, vector<Mat>& Ts, vector<Mat>& Rs, vector<string>& paths) {

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
	
		if (i!=1)
		{
			RTT[i] = RTs[1].inv()*RTs[i]* RTs[i];
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
		detail::SphericalWarper test_warp(panoSize.width  / M_PI/ 2.0);
		
		Mat image = imread(paths[i]);
		//Ks[1].convertTo(K1, CV_32FC1);
		Ks[i].convertTo(K1, CV_32FC1);
		point = test_warp.warp(image, K1, R, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
		points[i] = point;
// 		if (i==0)
// 		{
// 			point.x = point.x + 1342;
// 			point.y += 400;
// 		}
// 		else if (i == 2) {
// 		
// 			point.x -= 1342;
// 			point.y += 400;
// 		}
// 		else
// 		{
// 			point.y += 500;
// 		}

		cout << i << '\t' << point << endl;
		

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

}

//Omnidirectional--全向相机模型
bool OmniDModelPanoMap(cv::Mat &mapx, cv::Mat &mapy, Mat m_K, Mat m_D, Mat m_R, Mat m_T, Mat m_xi, int pano_width, int pano_height, double fov = 127.0, double radian_offset = M_PI / 2.0) {

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

	Mat Rtc, Rtl,RTT;

	Rtc= Common_GQ::toMat44(R_c, T_c).inv();
	Rtl = Common_GQ::toMat44(R_l, T_l).inv();
	RTT= Rtc.inv()*Rtl*Rtl;

	Mat R, T;
	Common_GQ::toMat33AndMat31(RTT, R, T);
	cout << "RX" << R << endl;
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
			
			Vec3d Xs(Xs_(0),Xs_(1),Xs_(2));
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

void OmniModel() {
// 	string path = "D:\\Data\\RobotCar\\camera_calibration\\OmniCamera\\Kl.yaml";
	
	//全景图映射表
	Mat pano_mapx, pano_mapy;
	//全景图的长
	int pano_width = 2048;
	//相机参数
	Mat K, D, R, T, xi;
	//加载参数
	LoadCameraParam(K, D, R, T, xi, 'c');
	cout << K << D << R << T << xi << endl;
	bool r = OmniDModelPanoMap(pano_mapx, pano_mapy, K, D,Mat::eye(3,3,CV_64FC1),T,xi, pano_width, pano_width / 2);
	if (!r)
	{
		cout << "Generate camera pano map failed!" << endl;
		return ;
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

//PTGui
bool PTGuiModel() {
	return true;
}

void extriStitch(const vector <Mat> &Ks, const vector<Mat>& Ds, vector<Mat>& Ts, vector<Mat>& Rs, vector<string>& paths) {

	
	vector<Mat> RTs(NUM_IMG), RTT(NUM_IMG), RT_dst(NUM_IMG);
	vector<Mat> imgs(NUM_IMG);
	Mat R, T, K1;
	for (size_t i = 0; i < NUM_IMG; i++)
	{
		RTs[i] = Common_GQ::toMat44(Rs[i], Ts[i]).inv();
		cout << "RTS:" << RTs[i] << endl;
		imgs[i] = imread(paths[i]);
	}


}

int main() {
	
	
	//图片数量
	int img_num = 3;

	string path1 = "D:/Data/RobotCar/camera_calibration/intrinsic/KD.yaml";
	string path2 = "D:/Data/RobotCar/camera_calibration/extrinsic/RT.yaml";

	vector<Mat> Ks, Rs, Ds, Ts, RTs(3), RTT(3),RT_dst(3);
	Mat K, D, R, T;
	initPara(K, D, R, T, 'l',path2, path1);
	Ks.emplace_back(K.clone());
	Rs.emplace_back(R.clone());
	Ds.emplace_back(D.clone());
	Ts.emplace_back(T.clone());
	initPara(K, D, R, T, 'c', path2, path1);
	Ks.emplace_back(K.clone());
	Rs.emplace_back(R.clone());
	Ds.emplace_back(D.clone());
	Ts.emplace_back(T.clone());

	initPara(K, D, R, T, 'r', path2, path1);
	Ks.emplace_back(K.clone());
	Rs.emplace_back(R.clone());
	Ds.emplace_back(D.clone());
	Ts.emplace_back(T.clone());

	vector<string> paths,path11;
	paths.push_back("D:\\Data\\RobotCar\\src\\1403537938670876_l.png");
	paths.push_back("D:\\Data\\RobotCar\\src\\1403537938670876_c.png");
	paths.push_back("D:\\Data\\RobotCar\\src\\1403537938670876_r.png");


	path11.push_back("D:\\Data\\RobotCar\\src\\left_undistort.png");
	path11.push_back("D:\\Data\\RobotCar\\src\\center_undistort.png");
	path11.push_back("D:\\Data\\RobotCar\\src\\right_undistort.png");


	//OriginModel(Ks, Ds, Ts, Rs, path11);
	//FishEyeModel(Ks, Ds, Ts, Rs, paths);

	OmniModel();






// 	vector<Mat> panos(3);
// 	vector<Mat> warped_images(3);
// 	vector<Point> points(3);
// 	vector<Mat> masks(3);
// 
// 	for (size_t i = 0; i < 3; i++)
// 	{
// 		Rs[i].convertTo(Rs[i], CV_32FC1);
// 		Ts[i].convertTo(Ts[i], CV_32FC1);
// 		RTs[i] = Common_GQ::toMat44(Rs[i], Ts[i]).inv();
// 		
// 		cout <<"RTS:"<< RTs[i] << endl;
// 	}
// 	Mat K1;
// 	Mat mapx1, mapy1;
// 	fisheye::estimateNewCameraMatrixForUndistortRectify(Ks[1], Ds[1], Size(1024, 1024), Mat::eye(3, 3, CV_64FC1), K1, 1.0, Size(1024, 1024));
// 	fisheye::initUndistortRectifyMap(Ks[1], Ds[1], Mat::eye(3, 3, CV_64FC1), K1, Size(1024, 1024), CV_32FC1, mapx1, mapy1);
// 
// //#pragma omp parallel for	
// 	for (size_t i = 0; i < 3; i++)
// 	{
// // 		Rs[i] = Rs[1].inv()*Rs[i];
// 		//RTT[i] = RTs[1].inv()*RTs[i];
// 		//cout <<"RTT:"<< RTT[i] << endl;
// 		
// 		if (i!=1)
// 		{
// 			RT_dst[i] = RTs[1].inv()*RTs[i] * RTs[i];
// 		}
// 		else
// 		{
// 			RT_dst[i] = Mat::eye(4, 4, CV_32FC1);
// 		}
// 		
// 		Common_GQ::toMat33AndMat31(RT_dst[i], R, T);
// 		cout << "RR:" << R << endl;
// 		
// 
// 		Mat img_test = imread(paths[i], -1);
// 		cvtColor(img_test, img_test, CV_BayerRG2RGB);
// 
// 		cv::Size img_size(img_test.cols, img_test.rows), undis_size(img_size * 4);
// 
// 		Mat warped_image;
// 		Mat pano_rgb(panoSize, CV_8UC3, 0.0);
// 		Point point;
// 
// 		Mat K_new, mapx, mapy, img_undis, pano_img;
// 
// 		fisheye::estimateNewCameraMatrixForUndistortRectify(Ks[i], Ds[i], img_size, Mat::eye(3, 3, CV_64FC1), K_new, 1.0, img_size);
// 		fisheye::initUndistortRectifyMap(Ks[i], Ds[i], Mat::eye(3, 3, CV_64FC1), K_new, img_size, CV_32FC1, mapx, mapy);
// 		remap(img_test, img_undis, mapx, mapy, CV_INTER_CUBIC);
// 
// 		detail::SphericalWarper test_warp(panoSize.width / 2.0 /M_PI);
// 		K_new.convertTo(K_new, CV_32FC1);
// 		K1.convertTo(K1, CV_32FC1);
// 
// 
// 		Ks[1].convertTo(Ks[1], CV_32FC1);
// 		Mat ll = imread(path11[i]);
// 		cout <<"K1::"<< Ks[1] << endl;
// 		point = test_warp.warp(img_undis, K_new, R, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
// 
// 		cout << i <<'\t'<< point<< endl;
// 		cout << T << endl;
// 
// 
// 	}


// 	Mat K1, D1, R1, T1;
// 	initPara(K1, D1, R1, T1, 'c', path2, path1);
// 	cout << R1 << endl;
// 	Ks.emplace_back(K1);
// 	Mat K2, D2, R2, T2;
// 	initPara(K2, D2, R2, T2, 'r', path2, path1);
// 	cout << R2 << endl;
// 	Ks.emplace_back(K2);

	//将左右相机统一到中间相机坐标系中
#if 0
	invert(R1, R1);
	R = R1*R;
	//中间相机使用单位矩阵投影
	double data[] = {1,0,0,0,1,0,0,0,1};
	Mat RR(3, 3, CV_64FC1, data);
	R2 = R1*R2;

	Rs.emplace_back(R);
	Rs.emplace_back(RR);
	Rs.emplace_back(R2);

	R.convertTo(R, CV_32FC1);
	return 0;

	Mat img;
	vector<Mat> images;
	vector<Mat> warped_images(img_num);

	img = imread("D:/Data/RobotCar/src/left_undistort.png");
	images.emplace_back(img);
	img = imread("D:/Data/RobotCar/src/center_undistort.png");
	images.emplace_back(img);
	img = imread("D:/Data/RobotCar/src/right_undistort.png");
	images.emplace_back(img);


	cout << Ks.size() << endl << images.size() << endl;
	for (int j = 0; j < img_num; j++)
	{
		//保证初始角度不变
		Mat init_R = Rs[j];
		init_R = init_R*Rs[j].inv();
		for (int i = 0; i < img_num; i++)
		{
			Rs[i] = init_R * Rs[i];
		}
	}


	detail::SphericalWarper swp(pano.cols / 2.0 / PI);
	Mat KK = Ks[1];
	for (int i = 0; i < img_num; i++)
	{
		Mat K,R;
		KK.convertTo(K, CV_32F);
		//Ks[i].convertTo(K, CV_32F);
		Rs[i].convertTo(R, CV_32F);
		cout <<"\nK:"<< K << endl <<"R:"<< R<<endl;
		Point point;
		Mat warped_image;
		Mat pano_rgb(panoSize, CV_8UC3, 0.0);
		point = swp.warp(images[i], K, R, CV_INTER_AREA, cv::BORDER_CONSTANT, warped_image);
		cout<<"\nPoint:" << point<<endl;

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
	//imwrite("pano.jpg", pano);
	//system("pano.jpg");
#endif

	return 0;
}