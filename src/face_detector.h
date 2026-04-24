#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include "camera_handler.h"
#include "channel.h"
#include <atomic>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <vector>

namespace facedet
{
  struct FacePose
  {
    cv::Rect bbox;
    std::vector<cv::Point2f> landmarks_68;
    std::vector<cv::Point2f> axis_points;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
  };

  struct FaceResult
  {
    std::vector<FacePose> faces;
  };

  struct FaceFrame
  {
    camh::CameraFrame camera;
    FaceResult result;
  };

  class FaceDetector
  {
  public:
    FaceDetector(utils::Channel<camh::CameraFrame>& input, const std::string& cascade_path, const std::string& lbf_model_path, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale);
    ~FaceDetector();

    const cv::Mat& CameraMatrix() const;
    utils::Channel<FaceFrame>& Frames();
    void SetEnabled(bool enabled);
    void Stop();

  private:
    struct Impl;
    void Run();

    utils::Channel<camh::CameraFrame>& input_;
    utils::Channel<FaceFrame> frames_;
    std::unique_ptr<Impl> impl_;
    std::atomic<bool> enabled_;
    std::thread worker_;
  };
} // namespace facedet

#endif // FACE_DETECTOR_H
