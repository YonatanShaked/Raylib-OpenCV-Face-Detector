#include "camera_handler.h"
#include <iostream>

namespace
{
  bool TryOpen(cv::VideoCapture& cap, int device_index, int api)
  {
    cap.release();

    if (api >= 0)
      return cap.open(device_index, api);

    return cap.open(device_index);
  }
} // namespace

namespace camh
{
  CameraHandler::CameraHandler(int device_index, int requested_width, int requested_height, int requested_fps)
    : cap_()
    , width_(0)
    , height_(0)
  {
    bool opened = false;

    opened = TryOpen(cap_, device_index, cv::CAP_V4L2);
    if (!opened)
      opened = TryOpen(cap_, device_index, cv::CAP_ANY);

    if (!opened)
    {
      std::cerr << "Could not open camera device " << device_index << "\n";
      return;
    }

    if (requested_width > 0)
      cap_.set(cv::CAP_PROP_FRAME_WIDTH, requested_width);

    if (requested_height > 0)
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, requested_height);

    if (requested_fps > 0)
      cap_.set(cv::CAP_PROP_FPS, requested_fps);

    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    width_ = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    height_ = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cerr << "Camera opened. Backend=" << cap_.get(cv::CAP_PROP_BACKEND) << " WxH=" << width_ << "x" << height_ << " FPS=" << cap_.get(cv::CAP_PROP_FPS) << "\n";
  }

  CameraHandler::~CameraHandler()
  {
    if (cap_.isOpened())
      cap_.release();
  }

  bool CameraHandler::IsOpened() const
  {
    return cap_.isOpened();
  }

  int CameraHandler::Width() const
  {
    return width_;
  }

  int CameraHandler::Height() const
  {
    return height_;
  }

  bool CameraHandler::Read(cv::Mat& out_bgr)
  {
    if (!cap_.isOpened())
      return false;

    cv::Mat frame;

    if (!cap_.read(frame))
      return false;

    if (frame.empty())
      return false;

    out_bgr = frame;
    return true;
  }
} // namespace camh