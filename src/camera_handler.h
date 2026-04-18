#ifndef CAMERA_HANDLER_H
#define CAMERA_HANDLER_H

#include "channel.h"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <thread>

namespace camh
{
  struct CameraFrame
  {
    std::uint64_t index;
    cv::Mat bgr;
  };

  class CameraHandler
  {
  public:
    CameraHandler(int device_index, int requested_width, int requested_height, int requested_fps);
    ~CameraHandler();

    bool IsOpened() const;
    int Width() const;
    int Height() const;
    utils::Channel<CameraFrame>& Frames();
    void Stop();

  private:
    void Run();

    cv::VideoCapture cap_;
    utils::Channel<CameraFrame> frames_;
    std::thread worker_;
    int width_;
    int height_;
  };
} // namespace camh

#endif // CAMERA_HANDLER_H
