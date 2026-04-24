#include "face_worker.h"

namespace facew
{
  FaceWorker::FaceWorker(utils::Channel<camh::CameraFrame>& input, const std::string& cascade_path, const std::string& lbf_model_path, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale)
    : input_(input)
    , frames_(2)
    , face_(cascade_path, lbf_model_path, image_width, image_height, max_faces, detect_every_n_frames, downscale)
    , enabled_(true)
    , worker_(&FaceWorker::Run, this)
  {
  }

  FaceWorker::~FaceWorker()
  {
    Stop();
  }

  const cv::Mat& FaceWorker::CameraMatrix() const
  {
    return face_.CameraMatrix();
  }

  utils::Channel<FaceFrame>& FaceWorker::Frames()
  {
    return frames_;
  }

  void FaceWorker::SetEnabled(bool enabled)
  {
    enabled_.store(enabled);
  }

  void FaceWorker::Stop()
  {
    input_.Close();
    frames_.Close();

    if (worker_.joinable())
      worker_.join();
  }

  void FaceWorker::Run()
  {
    camh::CameraFrame in;

    while (input_.Recv(in))
    {
      FaceFrame out;
      out.camera.index = in.index;
      out.camera.bgr = std::move(in.bgr);

      if (enabled_.load())
        out.result = face_.Process(out.camera.bgr);

      if (!frames_.Send(std::move(out)))
        break;
    }

    frames_.Close();
  }
} // namespace facew
