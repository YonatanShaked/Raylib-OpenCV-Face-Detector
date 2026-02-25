#ifndef RAYLIB_UTILS_H
#define RAYLIB_UTILS_H

#include <filesystem>
#include <opencv2/core.hpp>
#include <raylib.h>

namespace rlft
{
  std::filesystem::path AssetPath(const std::filesystem::path& rel);

  void DrawWebcamTexture(Texture2D tex, int img_w, int img_h, float& scale, float& off_x, float& off_y, float& draw_w, float& draw_h);
  Vector2 MapToWindow(const cv::Point2f& p, float scale, float off_x, float off_y);

  Camera3D MakeOpenCVCamera(const cv::Mat& K, int img_w, int img_h);

  bool RvecToAxisAngle(const cv::Vec3d& rvec, Vector3& out_axis, float& out_angle_deg);

  void DrawAxisBarsAtPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec, float len, float thick);
  void DrawGlassesAtPoseLit(Model& glasses, const cv::Vec3d& rvec, const cv::Vec3d& tvec);
} // namespace rlft

#endif // RAYLIB_UTILS_H
