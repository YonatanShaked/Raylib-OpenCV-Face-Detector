#include "face/detector.h"
#include "utils/asset_paths.h"
#include <algorithm>
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>

namespace
{
  utils::CameraIntrinsics MakeCameraIntrinsics(int w, int h)
  {
    utils::CameraIntrinsics intrinsics;
    intrinsics.fx = (double)w;
    intrinsics.fy = (double)w;
    intrinsics.cx = (double)w * 0.5;
    intrinsics.cy = (double)h * 0.5;
    return intrinsics;
  }

  cv::Mat MakeCameraMatrix(const utils::CameraIntrinsics& intrinsics)
  {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = intrinsics.fx;
    K.at<double>(1, 1) = intrinsics.fy;
    K.at<double>(0, 2) = intrinsics.cx;
    K.at<double>(1, 2) = intrinsics.cy;
    return K;
  }

  utils::Rect ToRect(const cv::Rect& rect)
  {
    utils::Rect out;
    out.x = rect.x;
    out.y = rect.y;
    out.width = rect.width;
    out.height = rect.height;
    return out;
  }

  utils::Point2f ToPoint2f(const cv::Point2f& point)
  {
    utils::Point2f out;
    out.x = point.x;
    out.y = point.y;
    return out;
  }

  utils::Vec3d ToVec3d(const cv::Vec3d& value)
  {
    utils::Vec3d out;
    out.x = value[0];
    out.y = value[1];
    out.z = value[2];
    return out;
  }

  class FaceCV
  {
  public:
    FaceCV(const std::string& cascade_path, const std::string& lbf_model_path, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale);

    face::FaceResult Process(const utils::ImageBuffer& bgr_frame);
    const utils::CameraIntrinsics& CameraIntrinsics() const;

  private:
    cv::CascadeClassifier face_cascade_;
    cv::Ptr<cv::face::Facemark> facemark_;

    utils::CameraIntrinsics camera_intrinsics_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    std::vector<cv::Point3d> object_points_;
    std::vector<int> object_point_ids_;

    int max_faces_;
    int detect_every_n_frames_;
    int downscale_;
    int frame_counter_;

    face::FaceResult last_result_;
  };

  FaceCV::FaceCV(const std::string& cascade_path, const std::string& lbf_model_path, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale)
    : camera_intrinsics_(MakeCameraIntrinsics(image_width, image_height))
    , camera_matrix_(MakeCameraMatrix(camera_intrinsics_))
    , dist_coeffs_(cv::Mat::zeros(5, 1, CV_64F))
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

  const utils::CameraIntrinsics& FaceCV::CameraIntrinsics() const
  {
    return camera_intrinsics_;
  }

  face::FaceResult FaceCV::Process(const utils::ImageBuffer& bgr_frame)
  {
    frame_counter_++;
    if (detect_every_n_frames_ > 1 && (frame_counter_ % detect_every_n_frames_) != 0)
      return last_result_;

    last_result_.faces.clear();

    if (bgr_frame.Empty() || bgr_frame.channels != 3)
      return last_result_;

    cv::Mat bgr(bgr_frame.height, bgr_frame.width, CV_8UC3, const_cast<std::uint8_t*>(bgr_frame.pixels.data()));
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
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
                return a.area() > b.area();
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

      face::FacePose pose;
      pose.bbox = ToRect(faces[i]);
      pose.landmarks_68.reserve(landmarks[i].size());
      for (const auto& landmark : landmarks[i])
        pose.landmarks_68.push_back(ToPoint2f(landmark));

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

      pose.rvec = ToVec3d(cv::Vec3d(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)));
      pose.tvec = ToVec3d(cv::Vec3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));

      std::vector<cv::Point3d> axis3d;
      double axis_len = 20.0;
      axis3d.push_back(cv::Point3d(0.0, 0.0, 0.0));
      axis3d.push_back(cv::Point3d(axis_len, 0.0, 0.0));
      axis3d.push_back(cv::Point3d(0.0, axis_len, 0.0));
      axis3d.push_back(cv::Point3d(0.0, 0.0, axis_len));

      std::vector<cv::Point2d> axis2d;
      cv::projectPoints(axis3d, rvec, tvec, camera_matrix_, dist_coeffs_, axis2d);

      pose.axis_points.reserve(axis2d.size());
      for (const auto& point : axis2d)
      {
        utils::Point2f axis_point;
        axis_point.x = (float)point.x;
        axis_point.y = (float)point.y;
        pose.axis_points.push_back(axis_point);
      }

      last_result_.faces.push_back(std::move(pose));
    }

    return last_result_;
  }
} // namespace

namespace face
{
  struct FaceDetector::Impl
  {
    explicit Impl(int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale)
      : face_(utils::AssetPath("haarcascade_frontalface_default.xml").string(), utils::AssetPath("lbfmodel.yaml").string(), image_width, image_height, max_faces, detect_every_n_frames, downscale)
    {
    }

    FaceCV face_;
  };

  FaceDetector::FaceDetector(utils::Channel<camera::CameraFrame>& input, int image_width, int image_height, int max_faces, int detect_every_n_frames, int downscale)
    : input_(input)
    , frames_(2)
    , impl_(std::make_unique<Impl>(image_width, image_height, max_faces, detect_every_n_frames, downscale))
    , enabled_(true)
    , worker_(&FaceDetector::Run, this)
  {
  }

  FaceDetector::~FaceDetector()
  {
    Stop();
  }

  const utils::CameraIntrinsics& FaceDetector::CameraIntrinsics() const
  {
    return impl_->face_.CameraIntrinsics();
  }

  utils::Channel<FaceFrame>& FaceDetector::Frames()
  {
    return frames_;
  }

  void FaceDetector::SetEnabled(bool enabled)
  {
    enabled_.store(enabled);
  }

  void FaceDetector::Stop()
  {
    input_.Close();
    frames_.Close();

    if (worker_.joinable())
      worker_.join();
  }

  void FaceDetector::Run()
  {
    camera::CameraFrame in;

    while (input_.Recv(in))
    {
      FaceFrame out;
      out.camera = std::move(in);

      if (enabled_.load())
        out.result = impl_->face_.Process(out.camera.bgr);

      if (!frames_.Send(std::move(out)))
        break;
    }

    frames_.Close();
  }
} // namespace face
