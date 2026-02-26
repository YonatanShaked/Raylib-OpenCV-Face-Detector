#ifndef FACE_CV_H
#define FACE_CV_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

namespace cvfd
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

  class FaceCV
  {
  public:
    FaceCV(const std::string& cascade_path, const std::string& lbf_model_path, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale);

    FaceResult Process(const cv::Mat& bgr_frame);

    int ImageWidth() const;
    int ImageHeight() const;

    const cv::Mat& CameraMatrix() const;

  private:
    cv::CascadeClassifier face_cascade_;
    cv::Ptr<cv::face::Facemark> facemark_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    std::vector<cv::Point3d> object_points_;
    std::vector<int> object_point_ids_;

    int img_w_;
    int img_h_;

    int max_faces_;
    int detect_every_n_frames_;
    int downscale_;
    int frame_counter_;

    FaceResult last_result_;
  };
} // namespace cvfd

#endif // FACE_CV_H