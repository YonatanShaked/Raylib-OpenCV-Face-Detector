#ifndef APP_H
#define APP_H

#include "face_detector.h"

namespace app
{
  void RunFaceTracker(facedet::FaceDetector& face_detector, int image_width, int image_height);
} // namespace app

#endif // APP_H
