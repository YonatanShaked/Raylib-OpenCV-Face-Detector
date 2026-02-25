#ifndef CAMERA_HANDLER_H
#define CAMERA_HANDLER_H

#include <opencv2/opencv.hpp>

class CameraHandler
{
public:
  CameraHandler(int device_index, int requested_width, int requested_height, int requested_fps);
  ~CameraHandler();

  bool IsOpened() const;
  int Width() const;
  int Height() const;

  bool Read(cv::Mat& out_bgr);

private:
  cv::VideoCapture cap_;
  int width_;
  int height_;
};

#endif // CAMERA_HANDLER_H
