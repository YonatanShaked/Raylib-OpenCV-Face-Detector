#ifndef APP_APP_H
#define APP_APP_H

#include "face/detector.h"

namespace app
{
  void RunFaceTracker(face::FaceDetector& face_detector, int image_width, int image_height);
} // namespace app

#endif // APP_APP_H
