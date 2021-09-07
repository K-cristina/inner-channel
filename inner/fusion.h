#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

class LaplacianBlending {
protected:
	int m_iLevel;
	cv::Mat m_blendedImg;
	cv::Mat sml(const cv::Mat& src);
	std::vector<cv::Mat> SMLPyramid(const cv::Mat& src);
	std::vector<cv::Mat> buildGaussianPyramid(const cv::Mat& img);
	std::vector<cv::Mat> buildLaplacianPyramid(const cv::Mat& img);
	cv::Mat collapsePyramid(std::vector<cv::Mat> inputPyrs);
	const cv::Mat xkernal = (cv::Mat_<float>(3, 3) << 0, 0, 0, -1, 2, -1, 0, 0, 0);
	const cv::Mat ykernal = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 2, 0, 0, -1, 0);
public:
	LaplacianBlending(int level);
	cv::Mat Result();
};

class in_Fusion :public LaplacianBlending
{
public:
	in_Fusion(int levels);
	cv::Mat Fusion(std::vector<cv::Mat>	I);

private:
	int radius = 15;
	double eps = 10e-4f;
	std::vector<cv::Mat> buildAddedPyramid(const cv::Mat& img);
	std::vector<cv::Mat> buildUpGaussianPyramid(const cv::Mat& img);
	void Index(const std::vector<cv::Mat>& input, cv::Mat& output);
};