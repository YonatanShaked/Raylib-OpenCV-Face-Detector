#include "app/app.h"
#include "assets/paths.h"
#include <cmath>
#include <raylib.h>
#include <rlgl.h>

#define RLIGHTS_IMPLEMENTATION
#include "rlights.h"

namespace
{
  bool RvecToAxisAngle(const vision::Vec3d& rvec, Vector3& out_axis, float& out_angle_deg)
  {
    double ax = rvec.x;
    double ay = rvec.y;
    double az = rvec.z;
    double angle = sqrt(ax * ax + ay * ay + az * az);
    if (angle < 1e-9)
      return false;

    out_axis = (Vector3){(float)(ax / angle), (float)(ay / angle), (float)(az / angle)};
    out_angle_deg = (float)(angle * 180.0 / 3.14159265358979323846);
    return true;
  }

  void ConvertBgrToRgba(const vision::ImageBuffer& src, vision::ImageBuffer& dst)
  {
    if (src.Empty() || src.channels != 3)
    {
      dst = {};
      return;
    }

    dst.width = src.width;
    dst.height = src.height;
    dst.channels = 4;
    dst.pixels.resize((size_t)src.width * (size_t)src.height * 4u);

    const std::uint8_t* src_ptr = src.pixels.data();
    std::uint8_t* dst_ptr = dst.pixels.data();
    const size_t pixel_count = (size_t)src.width * (size_t)src.height;
    for (size_t i = 0; i < pixel_count; i++)
    {
      dst_ptr[0] = src_ptr[2];
      dst_ptr[1] = src_ptr[1];
      dst_ptr[2] = src_ptr[0];
      dst_ptr[3] = 255;
      src_ptr += 3;
      dst_ptr += 4;
    }
  }

  Vector2 MapToWindow(const vision::Point2f& p, float scale, float off_x, float off_y)
  {
    Vector2 v;
    v.x = off_x + p.x * scale;
    v.y = off_y + p.y * scale;
    return v;
  }

  Camera3D MakePerspectiveCamera(const vision::CameraIntrinsics& intrinsics, int img_w, int img_h)
  {
    (void)img_w;
    double fovy = 2.0 * atan((double)img_h / (2.0 * intrinsics.fy));

    Camera3D cam;
    cam.position = (Vector3){0.0f, 0.0f, 0.0f};
    cam.target = (Vector3){0.0f, 0.0f, 1.0f};
    cam.up = (Vector3){0.0f, -1.0f, 0.0f};
    cam.fovy = (float)(fovy * 180.0 / 3.14159265358979323846);
    cam.projection = CAMERA_PERSPECTIVE;
    return cam;
  }

  void DrawWebcamTexture(Texture2D tex, int img_w, int img_h, float& scale, float& off_x, float& off_y, float& draw_w, float& draw_h)
  {
    int win_w = GetScreenWidth();
    int win_h = GetScreenHeight();

    scale = 1.0f;
    off_x = 0.0f;
    off_y = 0.0f;
    draw_w = (float)img_w;
    draw_h = (float)img_h;

    float sx = (float)win_w / (float)img_w;
    float sy = (float)win_h / (float)img_h;
    scale = (sx < sy) ? sx : sy;
    draw_w = (float)img_w * scale;
    draw_h = (float)img_h * scale;
    off_x = ((float)win_w - draw_w) * 0.5f;
    off_y = ((float)win_h - draw_h) * 0.5f;

    BeginDrawing();
    ClearBackground(BLACK);

    Rectangle src = {0.0f, 0.0f, (float)img_w, (float)img_h};
    Rectangle dst = {off_x, off_y, draw_w, draw_h};
    Vector2 origin = {0.0f, 0.0f};

    DrawTexturePro(tex, src, dst, origin, 0.0f, WHITE);
  }

  void DrawAxisBarsAtPose(const vision::Vec3d& rvec, const vision::Vec3d& tvec, float len, float thick)
  {
    Vector3 axis;
    float ang_deg = 0.0f;

    rlPushMatrix();
    rlTranslatef((float)tvec.x, (float)tvec.y, (float)tvec.z);

    if (RvecToAxisAngle(rvec, axis, ang_deg))
      rlRotatef(ang_deg, axis.x, axis.y, axis.z);

    DrawCubeV((Vector3){len * 0.5f, 0.0f, 0.0f}, (Vector3){len, thick, thick}, RED);
    DrawCubeV((Vector3){0.0f, len * 0.5f, 0.0f}, (Vector3){thick, len, thick}, GREEN);
    DrawCubeV((Vector3){0.0f, 0.0f, len * 0.5f}, (Vector3){thick, thick, len}, BLUE);

    rlPopMatrix();
  }

  void DrawModelAtPoseLit(Model& model, const vision::Vec3d& rvec, const vision::Vec3d& tvec)
  {
    Vector3 axis;
    float ang_deg = 0.0f;

    rlPushMatrix();
    rlTranslatef((float)tvec.x, (float)tvec.y, (float)tvec.z);

    if (RvecToAxisAngle(rvec, axis, ang_deg))
      rlRotatef(ang_deg, axis.x, axis.y, axis.z);

    model.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = (Color){15, 25, 70, 255};
    DrawModel(model, (Vector3){0.0f, 0.0f, 0.0f}, 1.0f, WHITE);

    rlPopMatrix();
  }
} // namespace

namespace app
{
  void RunFaceTracker(face::FaceDetector& face_detector, int image_width, int image_height)
  {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(image_width, image_height, "Raylib Face Tracker");
    SetTargetFPS(60);

    Image img = GenImageColor(image_width, image_height, BLACK);
    Texture2D tex = LoadTextureFromImage(img);
    UnloadImage(img);

    Model glasses_model = LoadModel(assets::AssetPath("glasses.obj").string().c_str());
    Shader light_shader = LoadShader(assets::AssetPath(std::filesystem::path("shaders") / "lighting.vs").string().c_str(), assets::AssetPath(std::filesystem::path("shaders") / "lighting.fs").string().c_str());

    for (int i = 0; i < glasses_model.materialCount; i++)
      glasses_model.materials[i].shader = light_shader;

    int loc_view_pos = GetShaderLocation(light_shader, "viewPos");
    Vector3 view_pos = {0.0f, 0.0f, 0.0f};
    SetShaderValue(light_shader, loc_view_pos, &view_pos.x, SHADER_UNIFORM_VEC3);

    Light light = CreateLight(LIGHT_DIRECTIONAL, (Vector3){0.0f, 0.0f, 0.0f}, (Vector3){0.3f, -0.7f, 1.0f}, WHITE, light_shader);

    vision::ImageBuffer frame_rgba;
    frame_rgba.width = image_width;
    frame_rgba.height = image_height;
    frame_rgba.channels = 4;
    frame_rgba.pixels.resize((size_t)image_width * (size_t)image_height * 4u, 0);

    Camera3D cv_cam = MakePerspectiveCamera(face_detector.CameraIntrinsics(), image_width, image_height);

    bool show_debug = false;
    bool do_cv = true;
    face::FaceResult faces;
    face::FaceFrame render_frame;
    bool has_frame = false;

    while (!WindowShouldClose())
    {
      if (IsKeyPressed(KEY_ONE))
        show_debug = !show_debug;

      if (IsKeyPressed(KEY_TWO))
      {
        do_cv = !do_cv;
        face_detector.SetEnabled(do_cv);
        if (!do_cv)
          faces.faces.clear();
      }

      face::FaceFrame next_frame;
      bool got_frame = false;
      while (face_detector.Frames().TryRecv(next_frame))
      {
        render_frame = std::move(next_frame);
        got_frame = true;
      }

      if (got_frame)
      {
        has_frame = true;
        ConvertBgrToRgba(render_frame.camera.bgr, frame_rgba);
        UpdateTexture(tex, frame_rgba.pixels.data());
        faces = render_frame.result;
      }

      float scale = 1.0f;
      float off_x = 0.0f;
      float off_y = 0.0f;
      float draw_w = 0.0f;
      float draw_h = 0.0f;
      DrawWebcamTexture(tex, image_width, image_height, scale, off_x, off_y, draw_w, draw_h);

      if (has_frame && do_cv)
      {
        BeginScissorMode((int)off_x, (int)off_y, (int)draw_w, (int)draw_h);
        rlViewport((int)off_x, (int)off_y, (int)draw_w, (int)draw_h);

        BeginMode3D(cv_cam);

        Vector3 vp = cv_cam.position;
        SetShaderValue(light_shader, loc_view_pos, &vp.x, SHADER_UNIFORM_VEC3);
        UpdateLightValues(light_shader, light);

        for (const auto& detected_face : faces.faces)
        {
          DrawModelAtPoseLit(glasses_model, detected_face.rvec, detected_face.tvec);
          if (show_debug)
            DrawAxisBarsAtPose(detected_face.rvec, detected_face.tvec, 15.0f, 1.0f);
        }

        EndMode3D();
        rlViewport(0, 0, GetScreenWidth(), GetScreenHeight());
        EndScissorMode();
      }

      if (has_frame && do_cv && show_debug)
      {
        for (const auto& detected_face : faces.faces)
        {
          Vector2 p1 = MapToWindow({(float)detected_face.bbox.x, (float)detected_face.bbox.y}, scale, off_x, off_y);
          Vector2 p2 = MapToWindow({(float)(detected_face.bbox.x + detected_face.bbox.width), (float)(detected_face.bbox.y + detected_face.bbox.height)}, scale, off_x, off_y);

          DrawRectangleLines((int)p1.x, (int)p1.y, (int)(p2.x - p1.x), (int)(p2.y - p1.y), RED);

          for (const auto& landmark : detected_face.landmarks_68)
          {
            Vector2 p = MapToWindow(landmark, scale, off_x, off_y);
            DrawCircleV(p, 2.0f, YELLOW);
          }
        }
      }

      DrawText(TextFormat("Press 1 to toggle debug info (%s)", show_debug ? "ON" : "OFF"), 10, 10, 20, GREEN);
      DrawText(TextFormat("Press 2 to toggle cv computations (%s)", do_cv ? "ON" : "OFF"), 10, 35, 20, GREEN);
      EndDrawing();
    }

    UnloadShader(light_shader);
    UnloadModel(glasses_model);
    UnloadTexture(tex);
    CloseWindow();
  }
} // namespace app
