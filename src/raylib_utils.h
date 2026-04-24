#ifndef RAYLIB_UTILS_H
#define RAYLIB_UTILS_H

#include "vision_types.h"
#include <raylib.h>

namespace rlft
{
  void DrawWebcamTexture(Texture2D tex, int img_w, int img_h, float& scale, float& off_x, float& off_y, float& draw_w, float& draw_h);
  void ConvertBgrToRgba(const vision::ImageBuffer& src, vision::ImageBuffer& dst);
  Vector2 MapToWindow(const vision::Point2f& p, float scale, float off_x, float off_y);
  Camera3D MakePerspectiveCamera(const vision::CameraIntrinsics& intrinsics, int img_w, int img_h);
  void DrawAxisBarsAtPose(const vision::Vec3d& rvec, const vision::Vec3d& tvec, float len, float thick);
  void DrawModelAtPoseLit(Model& model, const vision::Vec3d& rvec, const vision::Vec3d& tvec);
} // namespace rlft

#endif // RAYLIB_UTILS_H
