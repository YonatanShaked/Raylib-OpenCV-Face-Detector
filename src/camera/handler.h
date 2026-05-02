#ifndef CAMERA_HANDLER_H
#define CAMERA_HANDLER_H

#include "utils/channel.h"
#include "utils/vision_types.h"
#include <cstdint>
#include <memory>

namespace camera
{
  struct CameraFrame
  {
    std::uint64_t index;
    utils::ImageBuffer bgr;
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
    struct Impl;
    std::unique_ptr<Impl> impl_;
  };
} // namespace camera

#endif // CAMERA_HANDLER_H
