#include<iostream>
#include "fusion.h"

using namespace cv;
using namespace std;

int main()
{
	char name[200];
	vector<Mat>img, grayimg;
	Mat I, final_img;

	for (int i = 0; i < 7; i++)
	{
		sprintf_s(name, "C:/Users/Kuangtina/Desktop/comparision experiment/process/00%i.jpg", i);
		Mat pic = imread(name, IMREAD_UNCHANGED), oimg;
		/*cvtColor(pic, oimg, COLOR_RGB2GRAY);
		grayimg.push_back(oimg);*/
		assert(!pic.empty());
		img.push_back(pic);
	}
	
	in_Fusion matte(8);
	final_img = matte.Fusion(img);
	imshow("result", final_img);
	waitKey(0);
	return 0;
}