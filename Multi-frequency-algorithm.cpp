#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<iostream>
#include<vector>


using namespace std;
using namespace cv;
#define PI 3.1415926535

vector<Mat> image_all;

void decode_sinusoidal(cv::Mat mask, std::vector<cv::Mat> &srcImageSequence, cv::Mat &dstImage, int stepNum);
void decode_threefrequency(cv::Mat mask, std::vector<cv::Mat> &srcImageSequence, cv::Mat &unwrap);


int main()
{

	char name_gray[100];
	for (int i = 1; i< 25; i++)
	{
		sprintf_s(name_gray, "����3/%d.bmp", i);
		Mat image;
		image = imread(name_gray, 0);
		//std::vector<cv::Mat> channels;
		//cv::split(image, channels);
		//image_all.push_back(channels[0]);
		image_all.push_back(image);
	}
	cout << image_all.size() << endl;//����24���궨ͼ��

	int code_height = image_all[0].rows;
	int	code_weight = image_all[0].cols;

	Mat mask;
	mask = imread("mask.bmp", 0);


	vector<Mat> wrap_list_x;
	vector<Mat> wrap_list_y;

	for (int i = 0; i< 12; i += 4)
	{
		cv::Mat result;
		std::vector<cv::Mat> images;
		images.push_back(image_all[i + 0]);
		images.push_back(image_all[i + 1]);
		images.push_back(image_all[i + 2]);
		images.push_back(image_all[i + 3]);
		decode_sinusoidal(mask, images, result, 4);

		cv::Mat XsinWrapPhase = cv::Mat::zeros(code_height, code_weight, CV_64FC1);
		result.convertTo(XsinWrapPhase, CV_64FC1, 2 * PI, 0);
		wrap_list_x.push_back(XsinWrapPhase);
	}
	for (int i = 12; i< 24; i += 4)
	{
		cv::Mat result;
		std::vector<cv::Mat> images;
		images.push_back(image_all[i + 0]);
		images.push_back(image_all[i + 1]);
		images.push_back(image_all[i + 2]);
		images.push_back(image_all[i + 3]);
		decode_sinusoidal(mask, images, result, 4);

		cv::Mat YsinWrapPhase = cv::Mat::zeros(code_height, code_weight, CV_64FC1);
		result.convertTo(YsinWrapPhase, CV_64FC1, 2 * PI, 0);
		wrap_list_y.push_back(YsinWrapPhase);
	}
	cv::Mat unwrapMap_x;
	cv::Mat unwrapMap_y;

	decode_threefrequency(mask, wrap_list_x, unwrapMap_x);
	decode_threefrequency(mask, wrap_list_y, unwrapMap_y);


	Mat wrap_y1;
	wrap_y1 = unwrapMap_x.clone();
	Mat phase_image1 = Mat::zeros(code_height, code_weight, CV_8UC1);
	normalize(wrap_y1, wrap_y1, 1.0, 0.0, NORM_MINMAX);//��һ��0~1֮��
	wrap_y1.convertTo(phase_image1, CV_8UC1, 255, 0); //ת��Ϊ0~255֮�������
	namedWindow("����");
	imshow("����1", phase_image1);
	imwrite("2.bmp", phase_image1);

	for (int j = 0; j < 300; j++)
	{
		double* data = unwrapMap_x.ptr<double>(j);
		for (int i = 300; i< 301; i++)
		{
			cout << data[i] << "; ";
		}
	}

	waitKey(0);
	return 0;
}

void decode_sinusoidal(cv::Mat mask, std::vector<cv::Mat> &srcImageSequence, cv::Mat &dstImage, int stepNum)
{
	int ImageHeight = srcImageSequence[0].rows;
	int ImageWidth = srcImageSequence[0].cols;

	switch (stepNum) {
	case 3:
	{
		if (srcImageSequence.size()< 3)
		{
			std::cout << "the number of sinusoidal pattern is inadequate ... " << std::endl;
			return;
		}

		cv::Mat img1 = srcImageSequence.at(0);
		cv::Mat img2 = srcImageSequence.at(1);
		cv::Mat img3 = srcImageSequence.at(2);

		cv::Mat sinusoidal_phase = cv::Mat::zeros(ImageHeight, ImageWidth, CV_64FC1);
		for (int i = 0; i<ImageHeight; i++)
		{

			uchar* ptr1 = img1.ptr<uchar>(i);
			uchar* ptr2 = img2.ptr<uchar>(i);
			uchar* ptr3 = img3.ptr<uchar>(i);
			uchar* mask_ptr = mask.ptr<uchar>(i);
			double* optr = sinusoidal_phase.ptr<double>(i);

			for (int j = 0; j<ImageWidth; j++)
			{
				if (mask_ptr[j] != 0)
				{
					optr[j] = (double)atan2((double)(std::sqrt(3)*(ptr1[j] - ptr3[j])),
						(double)(2 * ptr2[j] - ptr1[j] - ptr3[j]));
				}
				else
				{
					optr[j] = 0;
				}
			}
		}

		cv::normalize(sinusoidal_phase, sinusoidal_phase, 1.0, 0.0, cv::NORM_MINMAX);

		dstImage = sinusoidal_phase.clone();
	}
	break;
	case 4:
	{
		if (srcImageSequence.size()< 4)
		{
			std::cout << "the number of sinusoidal pattern is inadequate ... " << std::endl;
			return;
		}

		cv::Mat img1 = srcImageSequence.at(0);
		cv::Mat img2 = srcImageSequence.at(1);
		cv::Mat img3 = srcImageSequence.at(2);
		cv::Mat img4 = srcImageSequence.at(3);

		cv::Mat sinusoidal_phase = cv::Mat::zeros(ImageHeight, ImageWidth, CV_64FC1);
		for (int i = 0; i<ImageHeight; i++)
		{

			uchar* ptr1 = img1.ptr<uchar>(i);
			uchar* ptr2 = img2.ptr<uchar>(i);
			uchar* ptr3 = img3.ptr<uchar>(i);
			uchar* ptr4 = img4.ptr<uchar>(i);
			uchar* mask_ptr = mask.ptr<uchar>(i);
			double* optr = sinusoidal_phase.ptr<double>(i);

			for (int j = 0; j<ImageWidth; j++)
			{
				if (mask_ptr[j] != 0)
				{
					optr[j] = (double)atan2((ptr4[j] - ptr2[j]),
						(ptr1[j] - ptr3[j]));
				}
				else
				{
					optr[j] = 0;
				}
			}
		}

		cv::normalize(sinusoidal_phase, sinusoidal_phase, 1.0, 0.0, cv::NORM_MINMAX);

		dstImage = sinusoidal_phase.clone();
	}
	break;
	default:
		break;
	}
}


void decode_threefrequency(cv::Mat mask, std::vector<cv::Mat> &srcImageSequence, cv::Mat &unwrap)
{
	cv::Mat image_1 = srcImageSequence.at(0);
	cv::Mat image_2 = srcImageSequence.at(1);
	cv::Mat image_3 = srcImageSequence.at(2);

	int nr = image_1.rows;
	int nc = image_1.cols;

	cv::Mat unwrap_comp_final_k1(nr, nc, CV_64FC1);

	//����Ƶ�ʲ�������
	float compensateBoundary = PI;

	float p1 = 52;
	float p2 = 58;
	float p3 = 620;
	float p12 = p2*p1 / (p2 - p1);
	float c3 = p12 / (p3 - p12);

	/***********************************************************************************************************************************/
	//������λ1��2����
	if (image_1.isContinuous())
	{
		if (image_2.isContinuous())
		{
			if (image_3.isContinuous())
			{
				if (mask.isContinuous())
				{
					nc = nc*nr;
					nr = 1;
				}
			}
		}
	}

	for (int i = 0; i< nr; i++)
	{
		double* ptr_w1 = image_1.ptr<double>(i);
		double* ptr_w2 = image_2.ptr<double>(i);
		double* ptr_w3 = image_3.ptr<double>(i);
		uchar *ptr_mask = mask.ptr<uchar>(i);
		double* ptr_un_comp_final_k1 = unwrap_comp_final_k1.ptr<double>(i);

		for (int j = 0; j< nc; j++)
		{
			if (ptr_mask[j] != 0)
			{
				//������λ1��2����
				double value_w12 = (ptr_w1[j] - ptr_w2[j]) / (2 * PI);
				if (value_w12< 0) {
					value_w12 += 1;
				}

				/***************************************************************************************/
				//�������λwrap_3��1��ֱ�Ӱ���������123-3��2�������kֵ�ٻ��㣬3�������ۺ�������
				double temp_unw3 = (value_w12 - ptr_w3[j] / (2 * PI));
				if (temp_unw3< 0) {
					temp_unw3 += 1;
				}

				//����չ��wrap3��СԾ��
				double value_unw3 = temp_unw3*c3 * 2 * PI;
				double k3 = floor(c3*temp_unw3);
				//kֵչ��wrap3,��Ծ��
				double value_un_k3 = k3 * 2 * PI + ptr_w3[j];

				//��ϱ���չ����kֵչ�����ֽ�����λ����comp_unwrap_3
				double value_un_err3 = value_un_k3 - value_unw3;

				if (compensateBoundary< value_un_err3)
				{
					value_un_k3 -= 2 * PI;
				}

				if (-compensateBoundary> value_un_err3)
				{
					value_un_k3 += 2 * PI;
				}

				double correct_unwrap3 = value_un_k3;

				//�����ϵ�unwrap1
				double value_unw1 = correct_unwrap3*p3 / p1;
				//У��
				double k1 = round((value_unw1 - ptr_w1[j]) / (2 * PI));

				ptr_un_comp_final_k1[j] = k1 * 2 * PI + ptr_w1[j];
				//ptr_un_comp_final_k1[j]=k1;
			}
			else
			{
				ptr_un_comp_final_k1[j] = 0;
			}
		}
	}
	unwrap = unwrap_comp_final_k1.clone();

}
