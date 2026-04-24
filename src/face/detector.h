#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include "camera/handler.h"
#include "utils/channel.h"
#include "vision/types.h"
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace face
{
  struct FacePose
  {
    vision::Rect bbox;
    std::vector<vision::Point2f> landmarks_68;
    std::vector<vision::Point2f> axis_points;
    vision::Vec3d rvec;
    vision::Vec3d tvec;
  };

  struct FaceResult
  {
    std::vector<FacePose> faces;
  };

  struct FaceFrame
  {
    camera::CameraFrame camera;
    FaceResult result;
  };

  class FaceDetector
  {
  public:
    FaceDetector(utils::Channel<camera::CameraFrame>& input, const std::string& cascade_path, const std::string& lbf_model_path, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale);
    ~FaceDetector();

    const vision::CameraIntrinsics& CameraIntrinsics() const;
    utils::Channel<FaceFrame>& Frames();
    void SetEnabled(bool enabled);
    void Stop();

  private:
    struct Impl;
    void Run();

    utils::Channel<camera::CameraFrame>& input_;
    utils::Channel<FaceFrame> frames_;
    std::unique_ptr<Impl> impl_;
    std::atomic<bool> enabled_;
    std::thread worker_;
  };
} // namespace face

#endif // FACE_DETECTOR_H
