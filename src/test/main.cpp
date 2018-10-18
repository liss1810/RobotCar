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
#include "Common_GQ.h"
#include "PanoStitch/panostitch.h"
#include <math.h>
#include "CornerFinder/CornerFinder.h"

#define _USE_MATH_DEFINES
#define PI 3.1415926
using namespace std;
using namespace cv;
using namespace detail;
using namespace Common_GQ;


int pano_w = 2000;
int pano_h = 1000;
//全景图大小
cv::Size panoSize(pano_w, pano_h);
//全景图
Mat pano(panoSize, CV_8UC3, Scalar::all(0));


bool initPara(Mat& K, Mat& D, Mat& R, Mat& T, char location, string filepath_rt, string filepath_kd) {

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


////////////////////////////////牛津robotcar数据集---鸟瞰图////////////////////////////////////

// 	int img_num = 3;
// 
// 	string path1 = "D:/Data/RobotCar/camera_calibration/intrinsic/KD.yaml";
// 	string path2 = "D:/Data/RobotCar/camera_calibration/extrinsic/RT.yaml";
// 
// 	vector<Mat> Ks, Rs, Ds, Ts, RTs(3), RTT(3), RT_dst(3);
// 	Mat K, D, R, T;
// 	initPara(K, D, R, T, 'l', path2, path1);
// 	Ks.emplace_back(K.clone());
// 	Rs.emplace_back(R.clone());
// 	Ds.emplace_back(D.clone());
// 	Ts.emplace_back(T.clone());
// 	initPara(K, D, R, T, 'c', path2, path1);
// 	Ks.emplace_back(K.clone());
// 	Rs.emplace_back(R.clone());
// 	Ds.emplace_back(D.clone());
// 	Ts.emplace_back(T.clone());
// 
// 	initPara(K, D, R, T, 'r', path2, path1);
// 	Ks.emplace_back(K.clone());
// 	Rs.emplace_back(R.clone());
// 	Ds.emplace_back(D.clone());
// 	Ts.emplace_back(T.clone());
// 
// 	vector<string> paths, path11;
// 	paths.push_back("D:/Data/RobotCar/src/1403537938670876_l.png");
// 	paths.push_back("D:/Data/RobotCar/src/1403537938670876_c.png");
// 	paths.push_back("D:/Data/RobotCar/src/1403537938670876_r.png");
// 
// 	PanoStitch ps;
// 	ps.BirdEyePano(Ks, Ds, Ts, Rs, paths);
	
////////////////////////////////牛津robotcar数据集/////////////////////////////////////////////

// 	int img_num = 3;
// 
// 	string path1 = "D:/Data/RobotCar/camera_calibration/intrinsic/KD.yaml";
// 	string path2 = "D:/Data/RobotCar/camera_calibration/extrinsic/RT.yaml";
// 
// 	vector<Mat> Ks, Rs, Ds, Ts, RTs(3), RTT(3),RT_dst(3);
// 	Mat K, D, R, T;
// 	initPara(K, D, R, T, 'l',path2, path1);
// 	Ks.emplace_back(K.clone());
// 	Rs.emplace_back(R.clone());
// 	Ds.emplace_back(D.clone());
// 	Ts.emplace_back(T.clone());
// 	initPara(K, D, R, T, 'c', path2, path1);
// 	Ks.emplace_back(K.clone());
// 	Rs.emplace_back(R.clone());
// 	Ds.emplace_back(D.clone());
// 	Ts.emplace_back(T.clone());
// 
// 	initPara(K, D, R, T, 'r', path2, path1);
// 	Ks.emplace_back(K.clone());
// 	Rs.emplace_back(R.clone());
// 	Ds.emplace_back(D.clone());
// 	Ts.emplace_back(T.clone());
// 
// 	vector<string> paths,path11;
// 	paths.push_back("D:/Data/RobotCar/src/1403537938670876_l.png");
// 	paths.push_back("D:/Data/RobotCar/src/1403537938670876_c.png");
// 	paths.push_back("D:/Data/RobotCar/src/1403537938670876_r.png");
// 
// 
// 	path11.push_back("D:/Data/RobotCar/src/left_undistort.png");
// 	path11.push_back("D:/Data/RobotCar/src/center_undistort.png");
// 	path11.push_back("D:/Data/RobotCar/src/right_undistort.png");
// 	PanoStitch ps;
// //  ps.OriginModel(Ks, Ds, Ts, Rs, path11);
// //	ps.FishEyeModel(Ks, Ds, Ts, Rs, paths);
// 	ps.FishModel_Test(Ks, Ds, Ts, Rs, paths);

////////////////////////////////全景拼接检验数据///////////////////////////////////////////////

// 	int num_pic = 8;
// 	vector<Mat> Rs(num_pic);
// 	string path_R = "D:/Data/Scene/R.yaml";
// 	string path_img = "D:/Data/Scene/scene4";
// 	Mat K = cv::Mat::eye(3, 3, CV_64FC1);
// 	K.at<double>(0, 0) = 1.8476431990924798e+003;
// 	K.at<double>(1, 1) = 1.8468263926957866e+003;
// 	K.at<double>(0, 2) = 3.0002819256452772e+003;
// 	K.at<double>(1, 2) = 2.0201214320738547e+003;
// 	double data[] = { -1.4075127613371857e-001, -8.9736194824215788e-003,4.8634042031003143e-003, 1.4425043608206586e-005 };
// 	Mat D(4, 1, CV_64FC1, data);
// 	cout << K << endl << D << endl;
// 	FileStorage fs(path_R, FileStorage::READ);
// 	stringstream ss;
// 	string str;
// 	for (int i = 0; i < num_pic; i++)
// 	{
// 		ss << "R" << i;
// 		ss >> str;
// 		fs[str] >> Rs[i];
// 		ss.clear();
// 	}
// 	PanoStitch ps;
// 	//ps.OmniSpherical(K, D, Rs, path_img, num_pic, panoSize);
// 	//ps.OpenCVSpherical(K, D, Rs, path_img, num_pic, panoSize, PanoStitch::CAM_FISHEYE);
// 
// 	ps.FishEyeModelSpherical(K, D, Rs, path_img, num_pic, panoSize);

///////////////////////////////利用PTGUI脚本拼接///////////////////////////////////////////////
	
// 	PanoStitch ps;
// 	char* pt_script = "D:/Data/RobotCar/ptgui/para.txt";
// 	string path_img = "D:/Data/RobotCar/ptgui";
// 	ps.PTGuiModel(pt_script, path_img);

///////////////////////////////Kinect实验采集数据//////////////////////////////////////////////

//相机个数
// 	int cam_num = 4;
// 	string path = "D:/Data/KinectCapture/cameraPara";
// 	vector<string> path_params(cam_num);
// 	vector<cameraPara> paras(cam_num);
// 	vector<cv::Mat> src_images(cam_num);
// 
// 	//读取图片
// 	string path_1 = "D:/LiuYongcan/VS2015/OpenCV/VehicleSurrounding/picture/001566361447/20.png";
// 	string path_2 = "D:/LiuYongcan/VS2015/OpenCV/VehicleSurrounding/picture/001597661447/20.png";
// 	string path_3 = "D:/LiuYongcan/VS2015/OpenCV/VehicleSurrounding/picture/001777653647/20.png";
// 	string path_4 = "D:/LiuYongcan/VS2015/OpenCV/VehicleSurrounding/picture/002515153647/20.png";
// 	src_images[0] = imread(path_1);
// 	src_images[1] = imread(path_2);
// 	src_images[2] = imread(path_3);
// 	src_images[3] = imread(path_4);
// 
// 	//获取相机参数
// 	path_params = Common_GQ::GetListFiles(path, "yaml");
// 	for (int i = 0; i < cam_num; i++)
// 	{
// 		//读取相机参数
// 		FileStorage fs(path_params[i], FileStorage::READ);
// 		cameraPara cp;
// 		fs["K"] >> cp.K;
// 		fs["D"] >> cp.D;
// 		//apriltag到相机的外参
// 		fs["RT1"] >> cp.RT;
// 		cp.index = i+1;
// 		paras[i] = cp;
// 	}
// 
// 	PanoStitch ps;
// 	float tag_len = 22.97;//cm
// 	ps.ApriltagModel(src_images, paras, tag_len);

//检验旋转相机，只有相机的光心在同一个圆上才能利用外参配准
// 	int cam_num = 2;
//  	string path = "D:/Data/KinectTest_9-30/test";
// 	string path_para = "D:/Data/KinectTest_9-30/test/intrinsic.yaml";
// 	vector<Mat> Ks(cam_num);
// 	vector<Mat> Rs(cam_num);
// 
// 	//
// 	FileStorage fs(path_para, FileStorage::READ);
// 	fs["K1"] >> Ks[0];
// 	fs["K2"] >> Ks[1];
// 
// 	double phi = M_PI / 4.0;
// 	double data1[] = { 1,0,0,0,1,0,0,0,1 };
// 	double data2[] = { cos(phi),0,sin(phi),0,1,0,-sin(phi),0,cos(phi) };
// 
// 	Rs[0] = Mat(3, 3, CV_64FC1, data1);
// 	Rs[1] =  Mat(3, 3, CV_64FC1, data2);
// 	cout << Rs[1] << endl;
// 
// 	PanoStitch ps;
// 	ps.KinectTest(Ks,Mat(3,3,CV_32FC1),Rs,path,cam_num,panoSize);


	int cam_num = 1;
	string path_img = "D:/Data/KinectTest/doubleKinect/001777653647/3.png";
	//string path_img = "D:/Data/KinectTest/doubleKinect/002515153647/3.png";
	string path_para = "D:/Data/KinectTest_9-30/test/intrinsic.yaml";
	cv::Mat K, D;

	cv::Mat img = imread(path_img);
	vector<cv::Point2f> corners;
	cv::Mat res;
	CornerFinder cf;
	cf.FindCornersOfAutoFixMissingCorners(img, corners, res);

	FileStorage fs(path_para, FileStorage::READ);
	fs["K3"] >> K;
	fs["D3"] >> D;
	
	cv::Mat dst;
	PanoStitch ps;
	ps.BirdEyeKinectTest(K, D, corners, img, dst, cam_num, 9, 8);

	return 0;
}