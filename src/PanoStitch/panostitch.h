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

//相机参数
typedef struct cameraPara
{
	int index;
	cv::Mat K;
	cv::Mat D;
	//到相机坐标系的外参
	cv::Mat RT;

}cameraPara;

class PanoStitch
{
public:

	//相机类型
	enum CAM_TYPE
	{
		CALIB_ONE_CAM,            //需要检校单个相机，所有图片为同个相机拍摄，适用于单个相机旋转拍静态全景
		CALIB_MULTI_CAM,          //需要检校多个相机，所有图片为多个相机同时拍摄，适用于多个相机定点拍全景
		CALIB_PTGUI               //通过PTGUI软件的特征点估计生成的包含相机内外参的PT Script文件做输入
	};

	enum CAM_MODEL
	{
		CAM_NOMAL,                //针孔模型
		CAM_FISHEYE,              //鱼眼模型
		CAM_OMNI				  //全向相机模型
	};
	

	PanoStitch();
	~PanoStitch();

	static bool LoadCameraParam(Mat& K, Mat& D, Mat& R, Mat& T, Mat &xi, char location, std::string filepath_rt = "D:/Data/RobotCar/camera_calibration/extrinsic/RT.yaml", std::string filepath_kd = "D:/Data/RobotCar/camera_calibration/OmniCamera/KD.yaml");
	/**
	 * 计算warp图的掩码图片
	 */
	bool ComputeImageMask(const cv::Mat &image, cv::Mat &imageMask, int dilateSize = -1, int erodeSize = -1);

	bool OmniSpherical(const cv::Mat K, const cv::Mat D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize, double fov=127.0, double radian_offset= 0*M_PI / 2.0);
	//
	bool FishEyeModelSpherical(const cv::Mat K, const cv::Mat D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize, double fov = 127.0, double radian_offset = 0 * M_PI / 2.0);
	
	bool OpenCVSpherical(const cv::Mat K, const cv::Mat D, const std::vector<cv::Mat> Rs,const std::string paths, int num_pic, cv::Size panoSize,CAM_MODEL model);
	
	//Omnidirectional--全向相机模型
	bool OmniDModelPanoMap(cv::Mat &mapx, cv::Mat &mapy, Mat m_K, Mat m_D, Mat m_R, Mat m_T, Mat m_xi, int pano_width, int pano_height, double fov = 127.0, double radian_offset = M_PI / 2.0);
	void OmniModel();

	//鱼眼----会使图片失真
	void FishEyeModel(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths);
	//普通
	void OriginModel(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths);
	//PTGui脚本方法
	bool PTGuiModel(const char* ptgui_script_path, const std::string paths);

	bool FishModel_Test(const std::vector<cv::Mat>& Ks, const std::vector<cv::Mat>& Ds, std::vector<cv::Mat>& Ts, std::vector<cv::Mat>& Rs, std::vector<std::string>& paths);

	//帽子加权融合算法
	bool LinearBlending(CAM_TYPE type, std::vector<cv::Mat> &warped_imgs, const std::vector<cv::Mat> &masks, cv::Mat &blended_img, int index,Size m_pano_size);

	bool ApriltagModel(std::vector<cv::Mat> src_images,std::vector<cameraPara> paras,float markLength);

	//检验数据
	bool KinectTest(const std::vector<cv::Mat> K, const cv::Mat D, const std::vector<cv::Mat> Rs, const std::string paths, int num_pic, cv::Size panoSize);
	//Kinect鸟瞰图
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

