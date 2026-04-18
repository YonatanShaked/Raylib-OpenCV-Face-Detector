#ifndef FACE_WORKER_H
#define FACE_WORKER_H

#include "camera_handler.h"
#include "channel.h"
#include "face_cv.h"
#include <atomic>
#include <thread>

namespace facew
{
  struct FaceFrame
  {
    camh::CameraFrame camera;
    cvfd::FaceResult result;
  };

  class FaceWorker
  {
  public:
    FaceWorker(utils::Channel<camh::CameraFrame>& input,
               const std::string& cascade_path,
               const std::string& lbf_model_path,
               int image_width,
               int image_height,
               int max_faces,
               int detect_every_n_frames,
               int downscale);
    ~FaceWorker();

    const cv::Mat& CameraMatrix() const;
    utils::Channel<FaceFrame>& Frames();
    void SetEnabled(bool enabled);
    void Stop();

  private:
    void Run();

    utils::Channel<camh::CameraFrame>& input_;
    utils::Channel<FaceFrame> frames_;
    cvfd::FaceCV face_;
    std::atomic<bool> enabled_;
    std::thread worker_;
  };
} // namespace facew

#endif // FACE_WORKER_H
