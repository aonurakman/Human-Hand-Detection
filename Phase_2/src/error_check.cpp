#include "../include/project_header.h"


using namespace std;
using namespace cv;


bool no_input_error (std::string arg) {

	if( arg == "" ){
		cout << "ERROR: Input required but none provided..." << endl;
		return true;
	}
	
	return false;

}


bool image_read_error (cv::Mat img, std::string filename) {

	if(img.empty()){
		cout<< "Error opening image:" << endl;
		cout << filename << endl;
		return true;
	}
	return false;
	
}



