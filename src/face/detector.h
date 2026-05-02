#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include "camera/handler.h"
#include "utils/channel.h"
#include "utils/vision_types.h"
#include <memory>
#include <vector>

namespace face
{
  struct FacePose
  {
    utils::Rect bbox;
    std::vector<utils::Point2f> landmarks_68;
    std::vector<utils::Point2f> axis_points;
    utils::Vec3d rvec;
    utils::Vec3d tvec;
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
    FaceDetector(utils::Channel<camera::CameraFrame>& input, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale);
    ~FaceDetector();

    const utils::CameraIntrinsics& CameraIntrinsics() const;
    utils::Channel<FaceFrame>& Frames();
    void SetEnabled(bool enabled);
    void Stop();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
  };
} // namespace face

#endif // FACE_DETECTOR_H
