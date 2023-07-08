#include "../include/project_header.h"


using namespace std;
using namespace cv;



int main (int argc, char** argv) {

	vector<string> filenames = image_name_generator(".jpg", 1, 30, true);
	vector<string> input_text = txt_reader("../../Phase_1/out_boxes/output.txt");
	
	int input_code = stoi(input_text[0]);
	cout << "Processing Image #" + to_string(input_code) << endl;
	Mat img = image_reader(filenames, "../img/", to_string(input_code), false, 1);
	
	vector<vector<int>> boxes = extract_boxes(input_text);
	vector<Rect> bounding_boxes = rect_extractor(boxes);
	
	
	for (int i = 0; i < bounding_boxes.size(); i++) 
	{
	
	
		Mat src = img(bounding_boxes[i]);
		Mat mask;
		inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
		src.setTo(Scalar(0, 0, 0), mask);
		
		imshow("Black Background Image", src);
		
		Mat kernel = (Mat_<float>(3,3) <<
		              1,  1, 1,
		              1, -8, 1,
		              1,  1, 1); 
		Mat imgLaplacian;
		filter2D(src, imgLaplacian, CV_32F, kernel);
		Mat sharp;
		src.convertTo(sharp, CV_32F);
		Mat imgResult = sharp - imgLaplacian;
		
		imgResult.convertTo(imgResult, CV_8UC3);
		imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
		
		imshow( "New Sharped Image", imgResult );
		
		Mat bw;
		cvtColor(imgResult, bw, COLOR_BGR2GRAY);
		threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
		imshow("Binary Image", bw);
		
		Mat dist;
		distanceTransform(bw, dist, DIST_L2, 3);
		
		normalize(dist, dist, 0, 1.0, NORM_MINMAX);
		imshow("Distance Transform Image", dist);
		
		threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
		
		Mat kernel1 = Mat::ones(3, 3, CV_8U);
		dilate(dist, dist, kernel1);
		imshow("Peaks", dist);
		
		Mat dist_8u;
		dist.convertTo(dist_8u, CV_8U);
		
		vector<vector<Point> > contours;
		findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		
		Mat markers = Mat::zeros(dist.size(), CV_32S);
		
		for (size_t i = 0; i < contours.size(); i++)
		{
		    drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
		}
		
		circle(markers, Point(5,5), 3, Scalar(255), -1);
		Mat markers8u;
		markers.convertTo(markers8u, CV_8U, 10);
		imshow("Markers", markers8u);
		
		watershed(imgResult, markers);
		Mat mark;
		markers.convertTo(mark, CV_8U);
		bitwise_not(mark, mark);
		
		vector<Vec3b> colors;
		for (size_t i = 0; i < contours.size(); i++)
		{
		    int b = theRNG().uniform(0, 256);
		    int g = theRNG().uniform(0, 256);
		    int r = theRNG().uniform(0, 256);
		    colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
		}
		
		Mat dst = Mat::zeros(markers.size(), CV_8UC3);
		
		for (int i = 0; i < markers.rows; i++)
		{
		    for (int j = 0; j < markers.cols; j++)
		    {
		        int index = markers.at<int>(i,j);
		        if (index > 0 && index <= static_cast<int>(contours.size()))
		        {
		            dst.at<Vec3b>(i,j) = colors[index-1];
		        }
		    }
		}
		
		imshow("Final Result", dst);
		waitKey();
		
		imwrite("../out_masks/"+to_string(i)+".jpg", dst);
	
	
	
	}

	//namedWindow("Image");
	//imshow("Image", img);
	//waitKey(0);
		
	return 0;
	
}






