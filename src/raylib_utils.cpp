#include "raylib_utils.h"
#include <cmath>
#include <raylib.h>
#include <rlgl.h>

#define RLIGHTS_IMPLEMENTATION
#include "rlights.h"

namespace rlft
{
  std::filesystem::path ExeDir()
  {
    const char* dir = GetApplicationDirectory();
    if (!dir || !dir[0])
      return std::filesystem::current_path();
    return std::filesystem::path(dir);
  }

  std::filesystem::path AssetPath(const std::filesystem::path& rel)
  {
    return ExeDir() / "assets" / rel;
  }

  void ComputeLetterbox(int win_w, int win_h, int img_w, int img_h, float& out_scale, float& out_off_x, float& out_off_y, float& out_draw_w, float& out_draw_h)
  {
    float sx = (float)win_w / (float)img_w;
    float sy = (float)win_h / (float)img_h;
    out_scale = (sx < sy) ? sx : sy;
    out_draw_w = (float)img_w * out_scale;
    out_draw_h = (float)img_h * out_scale;
    out_off_x = ((float)win_w - out_draw_w) * 0.5f;
    out_off_y = ((float)win_h - out_draw_h) * 0.5f;
  }

  Vector2 MapToWindow(const cv::Point2f& p, float scale, float off_x, float off_y)
  {
    Vector2 v;
    v.x = off_x + p.x * scale;
    v.y = off_y + p.y * scale;
    return v;
  }

  Camera3D MakeOpenCVCamera(const cv::Mat& K, int img_w, int img_h)
  {
    double fy = K.at<double>(1, 1);
    double fovy = 2.0 * atan((double)img_h / (2.0 * fy));
    Camera3D cam;
    cam.position = (Vector3){0.0f, 0.0f, 0.0f};
    cam.target = (Vector3){0.0f, 0.0f, 1.0f};
    cam.up = (Vector3){0.0f, -1.0f, 0.0f};
    cam.fovy = (float)(fovy * 180.0 / 3.14159265358979323846);
    cam.projection = CAMERA_PERSPECTIVE;
    return cam;
  }

  bool RvecToAxisAngle(const cv::Vec3d& rvec, Vector3& out_axis, float& out_angle_deg)
  {
    double ax = rvec[0];
    double ay = rvec[1];
    double az = rvec[2];
    double angle = sqrt(ax * ax + ay * ay + az * az);
    if (angle < 1e-9)
      return false;
    out_axis = (Vector3){(float)(ax / angle), (float)(ay / angle), (float)(az / angle)};
    out_angle_deg = (float)(angle * 180.0 / 3.14159265358979323846);
    return true;
  }

  void DrawAxisBarsAtPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec, float len, float thick)
  {
    Vector3 axis;
    float ang_deg = 0.0f;

    rlPushMatrix();
    rlTranslatef((float)tvec[0], (float)tvec[1], (float)tvec[2]);

    if (RvecToAxisAngle(rvec, axis, ang_deg))
      rlRotatef(ang_deg, axis.x, axis.y, axis.z);

    DrawCubeV((Vector3){len * 0.5f, 0.0f, 0.0f}, (Vector3){len, thick, thick}, RED);
    DrawCubeV((Vector3){0.0f, len * 0.5f, 0.0f}, (Vector3){thick, len, thick}, GREEN);
    DrawCubeV((Vector3){0.0f, 0.0f, len * 0.5f}, (Vector3){thick, thick, len}, BLUE);

    rlPopMatrix();
  }

  void DrawGlassesAtPoseLit(Model& glasses, const cv::Vec3d& rvec, const cv::Vec3d& tvec)
  {
    Vector3 axis;
    float ang_deg = 0.0f;

    rlPushMatrix();
    rlTranslatef((float)tvec[0], (float)tvec[1], (float)tvec[2]);

    if (RvecToAxisAngle(rvec, axis, ang_deg))
      rlRotatef(ang_deg, axis.x, axis.y, axis.z);

    rlRotatef(90.0f, 1.0f, 0.0f, 0.0f);
    rlRotatef(180.0f, 0.0f, 1.0f, 0.0f);

    glasses.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = (Color){15, 25, 70, 255};

    float s = 0.2f;
    DrawModel(glasses, (Vector3){0.0f, 0.0f, 0.0f}, s, WHITE);

    rlPopMatrix();
  }
}
