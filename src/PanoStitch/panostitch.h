#pragma once
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
#include <math.h>
//pano13
extern "C" {
#include <PTcommon.h>
};

#define NUM_IMG 3
#define _USE_MATH_DEFINES
using namespace cv;

//�������
typedef struct cameraPara
{
	int index;
	cv::Mat K;
	cv::Mat D;
	//���������ϵ�����
	cv::Mat RT;

}cameraPara;

class PanoStitch
{
public:

	//�������
	enum CAM_TYPE
	{
		CALIB_ONE_CAM,            //��Ҫ��У�������������ͼƬΪͬ��������㣬�����ڵ��������ת�ľ�̬ȫ��
		CALIB_MULTI_CAM,          //��Ҫ��У������������ͼƬΪ������ͬʱ���㣬�����ڶ�����������ȫ��
		CALIB_PTGUI               //ͨ��PTGUI�����������������ɵİ����������ε�PT Script�ļ�������
	};

	enum CAM_MODEL
	{
		CAM_NOMAL,                //���ģ��
		CAM_FISHEYE,              //����ģ��
		CAM_OMNI				  //ȫ�����ģ��
	};
	

	PanoStitch();
	~PanoStitch();

	static bool LoadCameraParam(Mat& K, Mat& D, Mat& R, Mat& T, Mat &xi, char location, std::string filepath_rt = "D:/Data/RobotCar/camera_calibration/extrinsic/RT.yaml", std::string filepath_kd = "D:/Data/RobotCar/camera_calibration/OmniCamera/KD.yaml");
	/**
	 * ����warpͼ������ͼƬ
	 */
	bool ComputeImageMask(const cv::Mat &image, cv::Mat &imageMask, int dilateSize = -1, int erodeSize = -1);

	bool OmniSpherical(const cv::Mat K, const cv::Mat D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize, double fov=127.0, double radian_offset= 0*M_PI / 2.0);
	//
	bool FishEyeModelSpherical(const cv::Mat K, const cv::Mat D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize, double fov = 127.0, double radian_offset = 0 * M_PI / 2.0);
	
	bool OpenCVSpherical(const cv::Mat K, const cv::Mat D, const std::vector<cv::Mat> Rs,const std::string paths, int num_pic, cv::Size panoSize,CAM_MODEL model);
	
	//Omnidirectional--ȫ�����ģ��
	bool OmniDModelPanoMap(cv::Mat &mapx, cv::Mat &mapy, Mat m_K, Mat m_D, Mat m_R, Mat m_T, Mat m_xi, int pano_width, int pano_height, double fov = 127.0, double radian_offset = M_PI / 2.0);
	void OmniModel();

	//����----��ʹͼƬʧ��
	void FishEyeModel(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths);
	//��ͨ
	void OriginModel(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths);
	//PTGui�ű�����
	bool PTGuiModel(const char* ptgui_script_path, const std::string paths);

	bool FishModel_Test(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths);

	//ñ�Ӽ�Ȩ�ں��㷨
	bool LinearBlending(CAM_TYPE type, std::vector<cv::Mat> &warped_imgs, const std::vector<cv::Mat> &masks, cv::Mat &blended_img, int index,Size m_pano_size);

	bool ApriltagModel(std::vector<cv::Mat> src_images,std::vector<cameraPara> paras,float markLength);

	//��������
	bool KinectTest(const std::vector<cv::Mat> K, const cv::Mat D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize);
	//Kinect���ͼ
	bool BirdEyeKinectTest(const cv::Mat K, const cv::Mat D, std::vector<cv::Point2f> corners, cv::Mat& src,cv::Mat&dst, int num_pic, int board_w, int board_h);
	//
	bool BirdEyePano(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths);
private:
	int pano_w;
	int pano_h;
	Size panoSize;
	Mat pano;
	Mat K;
	Mat D;
	Mat R;
	Mat T;
};

