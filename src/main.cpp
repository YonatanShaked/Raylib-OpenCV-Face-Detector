#include "camera_handler.h"
#include "face_detector.h"
#include "raylib_utils.h"
#include "rlights.h"
#include <raylib.h>
#include <rlgl.h>

int main(int argc, char** argv)
{
  std::filesystem::path cascade_path = rlft::AssetPath("haarcascade_frontalface_default.xml");
  std::filesystem::path lbf_path = rlft::AssetPath("lbfmodel.yaml");

  camh::CameraHandler cam(0, 1280, 720, 30);
  if (!cam.IsOpened())
    return 1;

  int img_w = cam.Width();
  int img_h = cam.Height();

  facedet::FaceDetector face(cam.Frames(), cascade_path.string(), lbf_path.string(), img_w, img_h, 5, 1, 1);

  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(img_w, img_h, "Raylib Face Tracker");
  SetTargetFPS(60);

  Image img = GenImageColor(img_w, img_h, BLACK);
  Texture2D tex = LoadTextureFromImage(img);
  UnloadImage(img);

  Model glasses_model = LoadModel(rlft::AssetPath("glasses.obj").string().c_str());

  Shader light_shader = LoadShader(rlft::AssetPath(std::filesystem::path("shaders") / "lighting.vs").string().c_str(), rlft::AssetPath(std::filesystem::path("shaders") / "lighting.fs").string().c_str());

  for (int i = 0; i < glasses_model.materialCount; i++)
    glasses_model.materials[i].shader = light_shader;

  int loc_view_pos = GetShaderLocation(light_shader, "viewPos");
  Vector3 view_pos = (Vector3){0.0f, 0.0f, 0.0f};
  SetShaderValue(light_shader, loc_view_pos, &view_pos.x, SHADER_UNIFORM_VEC3);

  Light light = CreateLight(LIGHT_DIRECTIONAL, (Vector3){0.0f, 0.0f, 0.0f}, (Vector3){0.3f, -0.7f, 1.0f}, WHITE, light_shader);

  cv::Mat frame_rgba = cv::Mat(img_h, img_w, CV_8UC4, cv::Scalar(0, 0, 0, 255));

  Camera3D cv_cam = rlft::MakeOpenCVCamera(face.CameraMatrix(), img_w, img_h);

  bool show_debug = false;
  bool do_cv = true;

  facedet::FaceResult fr;
  facedet::FaceFrame render_frame;
  bool has_frame = false;

  while (!WindowShouldClose())
  {
    if (IsKeyPressed(KEY_ONE))
      show_debug = !show_debug;

    if (IsKeyPressed(KEY_TWO))
    {
      do_cv = !do_cv;
      face.SetEnabled(do_cv);
      if (!do_cv)
        fr.faces.clear();
    }

    facedet::FaceFrame next_frame;
    bool got_frame = false;

    while (face.Frames().TryRecv(next_frame))
    {
      render_frame = std::move(next_frame);
      got_frame = true;
    }

    if (got_frame)
    {
      has_frame = true;
      cv::cvtColor(render_frame.camera.bgr, frame_rgba, cv::COLOR_BGR2RGBA);
      UpdateTexture(tex, frame_rgba.data);
      fr = render_frame.result;
    }

    float scale, off_x, off_y, draw_w, draw_h;
    rlft::DrawWebcamTexture(tex, img_w, img_h, scale, off_x, off_y, draw_w, draw_h);

    if (has_frame && do_cv)
    {
      BeginScissorMode((int)off_x, (int)off_y, (int)draw_w, (int)draw_h);
      rlViewport((int)off_x, (int)off_y, (int)draw_w, (int)draw_h);

      BeginMode3D(cv_cam);

      Vector3 vp = cv_cam.position;
      SetShaderValue(light_shader, loc_view_pos, &vp.x, SHADER_UNIFORM_VEC3);

      UpdateLightValues(light_shader, light);

      for (size_t fi = 0; fi < fr.faces.size(); fi++)
      {
        const auto& fp = fr.faces[fi];
        rlft::DrawModelAtPoseLit(glasses_model, fp.rvec, fp.tvec);

        if (show_debug)
          rlft::DrawAxisBarsAtPose(fp.rvec, fp.tvec, 15.0f, 1.0f);
      }

      EndMode3D();

      rlViewport(0, 0, GetScreenWidth(), GetScreenHeight());
      EndScissorMode();
    }

    if (has_frame && do_cv && show_debug)
    {
      for (size_t fi = 0; fi < fr.faces.size(); fi++)
      {
        const auto& fp = fr.faces[fi];

        Vector2 p1 = rlft::MapToWindow({(float)fp.bbox.x, (float)fp.bbox.y}, scale, off_x, off_y);
        Vector2 p2 = rlft::MapToWindow({(float)(fp.bbox.x + fp.bbox.width), (float)(fp.bbox.y + fp.bbox.height)}, scale, off_x, off_y);

        DrawRectangleLines((int)p1.x, (int)p1.y, (int)(p2.x - p1.x), (int)(p2.y - p1.y), RED);

        for (size_t i = 0; i < fp.landmarks_68.size(); i++)
        {
          Vector2 p = rlft::MapToWindow(fp.landmarks_68[i], scale, off_x, off_y);
          DrawCircleV(p, 2.0f, YELLOW);
        }
      }
    }

    DrawText(TextFormat("Press 1 to toggle debug info (%s)", show_debug ? "ON" : "OFF"), 10, 10, 20, GREEN);
    DrawText(TextFormat("Press 2 to toggle cv computations (%s)", do_cv ? "ON" : "OFF"), 10, 35, 20, GREEN);

    EndDrawing();
  }

  cam.Stop();
  face.Stop();

  UnloadShader(light_shader);
  UnloadModel(glasses_model);
  UnloadTexture(tex);
  CloseWindow();
  return 0;
}
