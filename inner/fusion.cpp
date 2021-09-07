#include "fusion.h"
#include <opencv2/ximgproc/edge_filter.hpp>

template<class T>
inline size_t argmax(T first, T last)
{
	return std::distance(first, std::max_element(first, last));
}

cv::Mat LaplacianBlending::sml(const cv::Mat & src)
{
	cv::Mat LX, LY, L;
	cv::filter2D(src, LX, -1, xkernal);
	cv::filter2D(src, LY, -1, ykernal);
	cv::Mat dst = abs(LX) + abs(LY);
	return dst;
}

std::vector<cv::Mat> LaplacianBlending::SMLPyramid(const cv::Mat & src)
{
	std::vector<cv::Mat> smlPyr;
	cv::Mat curImg = src, down, tmp;
	for (int i = 0; i < m_iLevel - 1; i++) {
		pyrDown(curImg, down);
		tmp = sml(curImg);
		smlPyr.push_back(tmp);
		curImg = down;
	}
	smlPyr.push_back(curImg);
	return smlPyr;
}

std::vector<cv::Mat> LaplacianBlending::buildGaussianPyramid(const cv::Mat & img)
{
	std::vector<cv::Mat> gaussPyr;
	gaussPyr.push_back(img); 
	cv::Mat curImg = img, tmp;
	for (int i = 1; i < m_iLevel; i++)
	{
		cv::pyrDown(curImg, tmp);
		gaussPyr.push_back(tmp);
		curImg = tmp;
	}
	return gaussPyr;
}

std::vector<cv::Mat> LaplacianBlending::buildLaplacianPyramid(const cv::Mat & img)
{
	std::vector<cv::Mat> lapPyr;
	cv::Mat curImg = img, down, up, tmp;
	for (int i = 0; i < m_iLevel; i++) {
		cv::pyrDown(curImg, down);
		cv::pyrUp(down, up, curImg.size());
		tmp = curImg - up;
		if (i < m_iLevel - 1)
			lapPyr.push_back(tmp);
		else
			lapPyr.push_back(curImg);
		curImg = down;
	}
	return lapPyr;
}

cv::Mat LaplacianBlending::collapsePyramid(std::vector<cv::Mat> inputPyrs)
{
	cv::Mat curImg = inputPyrs[m_iLevel - 1], tmp;
	for (int i = m_iLevel - 1; i > 0; i--) {
		cv::pyrUp(curImg, tmp, inputPyrs[i - 1].size());
		curImg = tmp + inputPyrs[i - 1];
	}
	return curImg;
}

LaplacianBlending::LaplacianBlending(int level) : m_iLevel(level)
{
}

cv::Mat LaplacianBlending::Result()
{
	return m_blendedImg;
}

in_Fusion::in_Fusion(int levels) :LaplacianBlending(levels)
{
}

cv::Mat in_Fusion::Fusion(std::vector<cv::Mat> I)
{
	int num = I.size(), n;
	cv::Mat pic, res, mask;
	std::vector<cv::Mat> fusionPyr(m_iLevel), tempPyr(num);
	std::vector<std::vector<cv::Mat>>imgPyr, addPyr;
	for (int i = 0; i < num; i++)
	{
		I[i].convertTo(pic, CV_32F);
		imgPyr.push_back(buildLaplacianPyramid(pic));
		addPyr.push_back(buildAddedPyramid(I[i]));
	}
	fusionPyr = imgPyr[0];
	for (int i = 0; i < m_iLevel; i++)
	{
		for (int j = 0; j < num; j++)
		{
			tempPyr[j] = addPyr[j][i];
		}
		Index(tempPyr, mask);
		for (size_t p = 0; p < imgPyr[0][i].rows; p++)
		{
			for (size_t q = 0; q < imgPyr[0][i].cols; q++)
			{
				n = (int)mask.at<uchar>(p, q);
				fusionPyr[i].at<cv::Vec3f>(p, q) = imgPyr[n][i].at<cv::Vec3f>(p, q);
			}
		}
	}
	res = collapsePyramid(fusionPyr);
	res.convertTo(res, CV_8U);
	return res;
}

std::vector<cv::Mat> in_Fusion::buildAddedPyramid(const cv::Mat & img)
{
	std::vector<cv::Mat> AddPyr(m_iLevel), smlPyr, depthPyr, HSV, TEM;
	cv::Mat pic, sparseMap, depthMap, temp, I;
	img.convertTo(I, CV_32F, 1.0 / 255);
	cv::cvtColor(I, pic, CV_RGB2HSV);
	cv::split(pic, HSV);	
	smlPyr = SMLPyramid(HSV[2]);
	sparseMap = sml(HSV[1]);
	cv::threshold(HSV[1], temp, 0, 1.0, cv::THRESH_BINARY);
	sparseMap += smlPyr[0].mul(1-temp);
	cv::ximgproc::guidedFilter(HSV[2], sparseMap, depthMap, radius, eps);
	depthPyr = buildGaussianPyramid(depthMap);
	for (int i = 0; i < m_iLevel; i++)
	{
		AddPyr[i] = smlPyr[i].mul(depthPyr[i]);
	}
	return AddPyr;
}

std::vector<cv::Mat> in_Fusion::buildUpGaussianPyramid(const cv::Mat & img)
{
	std::vector<cv::Mat> UpGPyr(m_iLevel);
	UpGPyr[m_iLevel - 1] = img;
	for (int i = m_iLevel - 1; i > 0; i--)
	{
		pyrUp(UpGPyr[i], UpGPyr[i - 1]);
	}
	return UpGPyr;
}

void in_Fusion::Index(const std::vector<cv::Mat>& input, cv::Mat & output)
{
	const int size = input.size(), row = input[0].rows, col = input[0].cols;
	std::vector<float> number(size);
	output = cv::Mat::zeros(input[0].size(), CV_8U);
	for (size_t i = 0; i < row; i++)
	{
		for (size_t j = 0; j < col; j++)
		{
			for (size_t n = 0; n < size; n++)
			{
				number[n] = input[n].ptr<float>(i)[j];
			}
			size_t index = argmax(number.begin(), number.end());
			output.ptr<char>(i)[j] = (int)(index);
		}
	}
}
