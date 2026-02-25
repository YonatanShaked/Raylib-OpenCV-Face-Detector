#ifndef RAYLIB_UTILS_H
#define RAYLIB_UTILS_H

#include <filesystem>
#include <opencv2/core.hpp>
#include <raylib.h>

namespace rlft
{
  std::filesystem::path ExeDir();
  std::filesystem::path AssetPath(const std::filesystem::path& rel);

  void ComputeLetterbox(int win_w, int win_h, int img_w, int img_h, float& out_scale, float& out_off_x, float& out_off_y, float& out_draw_w, float& out_draw_h);
  Vector2 MapToWindow(const cv::Point2f& p, float scale, float off_x, float off_y);

  Camera3D MakeOpenCVCamera(const cv::Mat& K, int img_w, int img_h);

  bool RvecToAxisAngle(const cv::Vec3d& rvec, Vector3& out_axis, float& out_angle_deg);

  void DrawAxisBarsAtPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec, float len, float thick);
  void DrawGlassesAtPoseLit(Model& glasses, const cv::Vec3d& rvec, const cv::Vec3d& tvec);
}

#endif // RAYLIB_UTILS_H
