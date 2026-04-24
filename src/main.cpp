#include "app/app.h"
#include "assets/paths.h"
#include "camera/handler.h"
#include "face/detector.h"

int main()
{
  std::filesystem::path cascade_path = assets::AssetPath("haarcascade_frontalface_default.xml");
  std::filesystem::path lbf_path = assets::AssetPath("lbfmodel.yaml");

  camera::CameraHandler cam(0, 1280, 720, 30);
  if (!cam.IsOpened())
    return 1;

  int img_w = cam.Width();
  int img_h = cam.Height();

  face::FaceDetector face_detector(cam.Frames(), cascade_path.string(), lbf_path.string(), img_w, img_h, 5, 1, 1);

  app::RunFaceTracker(face_detector, img_w, img_h);

  cam.Stop();
  face_detector.Stop();
  return 0;
}
