#include "app.h"
#include "asset_utils.h"
#include "camera_handler.h"
#include "face_detector.h"

int main(int argc, char** argv)
{
  std::filesystem::path cascade_path = assets::AssetPath("haarcascade_frontalface_default.xml");
  std::filesystem::path lbf_path = assets::AssetPath("lbfmodel.yaml");

  camh::CameraHandler cam(0, 1280, 720, 30);
  if (!cam.IsOpened())
    return 1;

  int img_w = cam.Width();
  int img_h = cam.Height();

  facedet::FaceDetector face(cam.Frames(), cascade_path.string(), lbf_path.string(), img_w, img_h, 5, 1, 1);

  app::RunFaceTracker(face, img_w, img_h);

  cam.Stop();
  face.Stop();
  return 0;
}
