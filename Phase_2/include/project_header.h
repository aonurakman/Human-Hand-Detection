#ifndef _PROJECT_HEADER_
#define _PROJECT_HEADER_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <string>
#include <iostream>


bool no_input_error (std::string arg);
bool image_read_error (cv::Mat img, std::string filename);

std::vector<std::string> image_name_generator(std::string format, int start, int end, bool zero_prefix);
std::vector<std::string> txt_reader(const std::string dir);
cv::Mat image_reader(const std::vector<std::string> filenames, const std::string dir, std::string arg, bool gray, float ratio);
std::vector<std::vector<int>> extract_boxes(std::vector<std::string> input_text);
std::vector<cv::Rect> rect_extractor(std::vector<std::vector<int>> boxes);


#endif
