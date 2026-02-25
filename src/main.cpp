#include "camera_handler.h"
#include "face_cv.h"
#include <string>
#include <raylib.h>
#include <rlgl.h>

#define RLIGHTS_IMPLEMENTATION
#include "rlights.h"

static void ComputeLetterbox(int win_w, int win_h, int img_w, int img_h, float& out_scale, float& out_off_x, float& out_off_y, float& out_draw_w, float& out_draw_h)
{
  float sx = (float)win_w / (float)img_w;
  float sy = (float)win_h / (float)img_h;
  out_scale = (sx < sy) ? sx : sy;
  out_draw_w = (float)img_w * out_scale;
  out_draw_h = (float)img_h * out_scale;
  out_off_x = ((float)win_w - out_draw_w) * 0.5f;
  out_off_y = ((float)win_h - out_draw_h) * 0.5f;
}

static Vector2 MapToWindow(const cv::Point2f& p, float scale, float off_x, float off_y)
{
  Vector2 v;
  v.x = off_x + p.x * scale;
  v.y = off_y + p.y * scale;
  return v;
}

static Camera3D MakeOpenCVCamera(const cv::Mat& K, int img_w, int img_h)
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

static bool RvecToAxisAngle(const cv::Vec3d& rvec, Vector3& out_axis, float& out_angle_deg)
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

static void DrawAxisBarsAtPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec, float len, float thick)
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

static void DrawGlassesAtPoseLit(Model& glasses, const cv::Vec3d& rvec, const cv::Vec3d& tvec)
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

static const char* OnOffText(bool v)
{
  return v ? "ON" : "OFF";
}

int main(int argc, char** argv)
{
  std::string cascade_path = "assets/haarcascade_frontalface_default.xml";
  std::string lbf_path = "assets/lbfmodel.yaml";

  CameraHandler cam(0, 1280, 720, 30);
  if (!cam.IsOpened())
    return 1;

  int img_w = cam.Width();
  int img_h = cam.Height();

  FaceCV face(cascade_path, lbf_path, img_w, img_h, 5, 1, 1);

  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(img_w, img_h, "raylib + opencv facemark");
  SetTargetFPS(60);

  Image img = GenImageColor(img_w, img_h, BLACK);
  Texture2D tex = LoadTextureFromImage(img);
  UnloadImage(img);

  Model glasses_model = LoadModel("assets/glasses.obj");

  Shader light_shader = LoadShader("assets/shaders/lighting.vs", "assets/shaders/lighting.fs");

  for (int i = 0; i < glasses_model.materialCount; i++)
    glasses_model.materials[i].shader = light_shader;

  int loc_view_pos = GetShaderLocation(light_shader, "viewPos");
  Vector3 view_pos = (Vector3){0.0f, 0.0f, 0.0f};
  SetShaderValue(light_shader, loc_view_pos, &view_pos.x, SHADER_UNIFORM_VEC3);

  Light light = CreateLight(LIGHT_DIRECTIONAL, (Vector3){0.0f, 0.0f, 0.0f}, (Vector3){0.3f, -0.7f, 1.0f}, WHITE, light_shader);

  cv::Mat frame_bgr;
  cv::Mat frame_rgba;

  Camera3D cv_cam = MakeOpenCVCamera(face.CameraMatrix(), img_w, img_h);

  bool show_debug = false;
  bool do_cv = true;

  FaceResult fr;

  while (!WindowShouldClose())
  {
    if (IsKeyPressed(KEY_ONE))
      show_debug = !show_debug;

    if (IsKeyPressed(KEY_TWO))
      do_cv = !do_cv;

    if (cam.Read(frame_bgr))
    {
      cv::cvtColor(frame_bgr, frame_rgba, cv::COLOR_BGR2RGBA);
      UpdateTexture(tex, frame_rgba.data);
    }

    if (do_cv)
      fr = face.Process(frame_bgr);
    else
      fr.faces.clear();

    int win_w = GetScreenWidth();
    int win_h = GetScreenHeight();

    float scale = 1.0f;
    float off_x = 0.0f;
    float off_y = 0.0f;
    float draw_w = (float)img_w;
    float draw_h = (float)img_h;

    ComputeLetterbox(win_w, win_h, img_w, img_h, scale, off_x, off_y, draw_w, draw_h);

    BeginDrawing();
    ClearBackground(BLACK);

    Rectangle src;
    src.x = 0.0f;
    src.y = 0.0f;
    src.width = (float)img_w;
    src.height = (float)img_h;

    Rectangle dst;
    dst.x = off_x;
    dst.y = off_y;
    dst.width = draw_w;
    dst.height = draw_h;

    Vector2 origin;
    origin.x = 0.0f;
    origin.y = 0.0f;

    DrawTexturePro(tex, src, dst, origin, 0.0f, WHITE);

    if (do_cv)
    {
      BeginScissorMode((int)off_x, (int)off_y, (int)draw_w, (int)draw_h);
      rlViewport((int)off_x, (int)off_y, (int)draw_w, (int)draw_h);

      BeginMode3D(cv_cam);

      Vector3 vp = cv_cam.position;
      SetShaderValue(light_shader, loc_view_pos, &vp.x, SHADER_UNIFORM_VEC3);

      UpdateLightValues(light_shader, light);

      for (size_t fi = 0; fi < fr.faces.size(); fi++)
        DrawGlassesAtPoseLit(glasses_model, fr.faces[fi].rvec, fr.faces[fi].tvec);

      EndMode3D();

      rlViewport(0, 0, win_w, win_h);
      EndScissorMode();

      if (show_debug)
      {
        for (size_t fi = 0; fi < fr.faces.size(); fi++)
        {
          const FacePose& fp = fr.faces[fi];

          DrawAxisBarsAtPose(fp.rvec, fp.tvec, 15.0f, 1.0f);

          float min_x = fp.landmarks_68[0].x;
          float max_x = fp.landmarks_68[0].x;
          float min_y = fp.landmarks_68[0].y;
          float max_y = fp.landmarks_68[0].y;

          for (size_t i = 1; i < fp.landmarks_68.size(); i++)
          {
            if (fp.landmarks_68[i].x < min_x)
              min_x = fp.landmarks_68[i].x;
            if (fp.landmarks_68[i].x > max_x)
              max_x = fp.landmarks_68[i].x;
            if (fp.landmarks_68[i].y < min_y)
              min_y = fp.landmarks_68[i].y;
            if (fp.landmarks_68[i].y > max_y)
              max_y = fp.landmarks_68[i].y;
          }

          Vector2 p1 = MapToWindow({min_x, min_y}, scale, off_x, off_y);
          Vector2 p2 = MapToWindow({max_x, max_y}, scale, off_x, off_y);

          DrawRectangleLines((int)p1.x, (int)p1.y, (int)(p2.x - p1.x), (int)(p2.y - p1.y), RED);

          for (size_t i = 0; i < fp.landmarks_68.size(); i++)
          {
            Vector2 p = MapToWindow(fp.landmarks_68[i], scale, off_x, off_y);
            DrawCircleV(p, 2.0f, YELLOW);
          }
        }
      }
    }

    DrawText(TextFormat("Press 1 to toggle debug info (%s)", OnOffText(show_debug)), 10, 10, 20, GREEN);
    DrawText(TextFormat("Press 2 to toggle cv computations (%s)", OnOffText(do_cv)), 10, 35, 20, GREEN);

    EndDrawing();
  }

  UnloadShader(light_shader);
  UnloadModel(glasses_model);
  UnloadTexture(tex);
  CloseWindow();
  return 0;
}