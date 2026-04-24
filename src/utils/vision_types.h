#ifndef UTILS_VISION_TYPES_H
#define UTILS_VISION_TYPES_H

#include <cstdint>
#include <vector>

namespace utils
{
  struct Point2f
  {
    float x = 0.0f;
    float y = 0.0f;
  };

  struct Rect
  {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
  };

  struct Vec3d
  {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
  };

  struct CameraIntrinsics
  {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
  };

  struct ImageBuffer
  {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<std::uint8_t> pixels;

    bool Empty() const
    {
      return pixels.empty() || width <= 0 || height <= 0 || channels <= 0;
    }
  };
} // namespace utils

#endif // UTILS_VISION_TYPES_H
