#include "face_cv.h"
#include <algorithm>

static cv::Mat MakeCameraMatrix(int w, int h)
{
  cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
  double f = (double)w;
  K.at<double>(0, 0) = f;
  K.at<double>(1, 1) = f;
  K.at<double>(0, 2) = (double)w * 0.5;
  K.at<double>(1, 2) = (double)h * 0.5;
  return K;
}

FaceCV::FaceCV(const std::string& cascade_path, const std::string& lbf_model_path, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale)
  : camera_matrix_(MakeCameraMatrix(image_width, image_height))
  , dist_coeffs_(cv::Mat::zeros(5, 1, CV_64F))
  , img_w_(image_width)
  , img_h_(image_height)
  , max_faces_(max_faces)
  , detect_every_n_frames_(detect_every_n_frames)
  , downscale_(downscale)
  , frame_counter_(0)
{
  face_cascade_.load(cascade_path);

  facemark_ = cv::face::FacemarkLBF::create();
  facemark_->loadModel(lbf_model_path);

  object_points_.push_back(cv::Point3d(8.27412, 1.33849, 10.63490));
  object_points_.push_back(cv::Point3d(-8.27412, 1.33849, 10.63490));
  object_points_.push_back(cv::Point3d(0.0, -4.47894, 17.73010));
  object_points_.push_back(cv::Point3d(-4.61960, -10.14360, 12.27940));
  object_points_.push_back(cv::Point3d(4.61960, -10.14360, 12.27940));

  object_point_ids_.push_back(45);
  object_point_ids_.push_back(36);
  object_point_ids_.push_back(30);
  object_point_ids_.push_back(48);
  object_point_ids_.push_back(54);
}

int FaceCV::ImageWidth() const
{
  return img_w_;
}

int FaceCV::ImageHeight() const
{
  return img_h_;
}

const cv::Mat& FaceCV::CameraMatrix() const
{
  return camera_matrix_;
}

FaceResult FaceCV::Process(const cv::Mat& bgr_frame)
{
  frame_counter_++;
  if (detect_every_n_frames_ > 1 && (frame_counter_ % detect_every_n_frames_) != 0)
    return last_result_;

  last_result_.faces.clear();

  if (bgr_frame.empty())
    return last_result_;

  cv::Mat gray;
  cv::cvtColor(bgr_frame, gray, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(gray, gray);

  cv::Mat gray_small = gray;
  float scale_up = 1.0f;

  if (downscale_ > 1)
  {
    cv::resize(gray, gray_small, cv::Size(gray.cols / downscale_, gray.rows / downscale_), 0, 0, cv::INTER_LINEAR);
    scale_up = (float)downscale_;
  }

  std::vector<cv::Rect> faces_small;
  face_cascade_.detectMultiScale(gray_small, faces_small, 1.1, 2, 0, cv::Size(30 / downscale_, 30 / downscale_));

  if (faces_small.empty())
    return last_result_;

  std::vector<cv::Rect> faces;
  faces.reserve(faces_small.size());
  for (const auto& r : faces_small)
  {
    cv::Rect rf;
    rf.x = (int)(r.x * scale_up);
    rf.y = (int)(r.y * scale_up);
    rf.width = (int)(r.width * scale_up);
    rf.height = (int)(r.height * scale_up);
    rf &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (rf.width > 0 && rf.height > 0)
      faces.push_back(rf);
  }

  std::sort(faces.begin(),
            faces.end(),
            [](const cv::Rect& a, const cv::Rect& b)
            {
              return (a.area() > b.area());
            });

  if ((int)faces.size() > max_faces_)
    faces.resize(max_faces_);

  std::vector<std::vector<cv::Point2f>> landmarks;
  bool ok = false;
  try
  {
    ok = facemark_->fit(gray, faces, landmarks);
  }
  catch (...)
  {
    ok = false;
  }

  if (!ok || landmarks.empty())
    return last_result_;

  int count = (int)landmarks.size();
  if (count > max_faces_)
    count = max_faces_;

  for (int i = 0; i < count; i++)
  {
    if (landmarks[i].size() < 55)
      continue;

    FacePose pose;
    pose.landmarks_68 = landmarks[i];
    pose.axis_points.clear();
    pose.rvec = cv::Vec3d(0.0, 0.0, 0.0);
    pose.tvec = cv::Vec3d(0.0, 0.0, 0.0);

    std::vector<cv::Point2d> image_points;
    image_points.reserve(object_point_ids_.size());
    for (size_t j = 0; j < object_point_ids_.size(); j++)
    {
      int idx = object_point_ids_[j];
      image_points.push_back(cv::Point2d(pose.landmarks_68[idx].x, pose.landmarks_68[idx].y));
    }

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

    bool pnp_ok = cv::solvePnP(object_points_, image_points, camera_matrix_, dist_coeffs_, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

    if (!pnp_ok)
      continue;

    pose.rvec = cv::Vec3d(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0));
    pose.tvec = cv::Vec3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));

    std::vector<cv::Point3d> axis3d;
    double axis_len = 20.0;
    axis3d.push_back(cv::Point3d(0.0, 0.0, 0.0));
    axis3d.push_back(cv::Point3d(axis_len, 0.0, 0.0));
    axis3d.push_back(cv::Point3d(0.0, axis_len, 0.0));
    axis3d.push_back(cv::Point3d(0.0, 0.0, axis_len));

    std::vector<cv::Point2d> axis2d;
    cv::projectPoints(axis3d, rvec, tvec, camera_matrix_, dist_coeffs_, axis2d);

    pose.axis_points.reserve(axis2d.size());
    for (size_t k = 0; k < axis2d.size(); k++)
      pose.axis_points.push_back(cv::Point2f((float)axis2d[k].x, (float)axis2d[k].y));

    last_result_.faces.push_back(pose);
  }

  return last_result_;
}