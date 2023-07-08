#include "../include/project_header.h"

using namespace std;
using namespace cv;



std::vector<std::string> image_name_generator(std::string format, int start, int end, bool zero_prefix){ 

	vector<string> filenames = {};
	for (int idx = start; idx <= end; idx++) {
		string f_name = "";
		if ((zero_prefix)&&(idx < 10)) {
			f_name = "0";
		}
		f_name += to_string(idx);
		f_name += format;
		filenames.push_back(f_name);
	}

	return filenames;

}


cv::Mat image_reader(const std::vector<std::string> filenames, const std::string dir, std::string arg, bool gray, float ratio) {

	Mat blank = Mat::zeros(100, 100, CV_8UC1);
	
	if ( no_input_error(arg) ) { return blank; }
	
	string filename = dir + filenames[stoi(arg)-1];
	
	
	Mat img;
	if (gray) { img = imread(filename, COLOR_BGR2GRAY); }
	else { img = imread(filename); }
	
	
	if ( image_read_error(img, filename)) { return blank; }
	
	resize(img, img, Size(), ratio, ratio);
	
	return img;
	
}



std::vector<std::string> txt_reader(const std::string dir) {

	vector<string> txt = {};
	ifstream newfile;
	newfile.open(dir, ios::in);
	
	if ( newfile.is_open() ) 
	{
	
	  string tp;
	  while(!newfile.eof()) {
	  	getline(newfile, tp);
	  	if (!tp.empty()) {
	  		txt.push_back(tp);
	  	}
	  }
	  newfile.close(); 
	  
	  
	} else { cout << "Error Reading: " + dir << endl;	}
	
	return txt;
	
}

std::vector<std::vector<int>> extract_boxes(std::vector<std::string> input_text) {

	vector<vector<int>> boxes = {};
	for (int i = 1; i < input_text.size(); i++ ) {	
		string box = input_text[i];
		string tmp = "";
		vector<int> vals_vec = {};
		for (int j = 0; j < box.size(); j++) {
			if ((int(box[j]) < 48) || (int(box[j]) > 57)) {
				int val = stoi(tmp);
				vals_vec.push_back(val);
				tmp = "";
			} else { tmp += box[j]; }	
		}
		
		int val = stoi(tmp);
		vals_vec.push_back(val);
		
		boxes.push_back(vals_vec);
	}
	
	return boxes;

}


std::vector<cv::Rect> rect_extractor(std::vector<std::vector<int>> boxes){


	std::vector<cv::Rect> bounding_boxes = {};
	for (int i = 0; i < boxes.size(); i++ ) {
		cv::Rect rect(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]);
		bounding_boxes.push_back(rect);
	}
	
	return bounding_boxes;

}












