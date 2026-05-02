#include "camera/handler.h"
#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

namespace
{
  constexpr GstClockTime kPullTimeout = 100 * GST_MSECOND;

  struct GstElementDeleter
  {
    void operator()(GstElement* element) const
    {
      if (!element)
        return;

      gst_element_set_state(element, GST_STATE_NULL);
      gst_object_unref(element);
    }
  };

  std::string MakeCaps(const char* media_type, int requested_width, int requested_height, int requested_fps)
  {
    std::ostringstream caps;
    caps << media_type;

    if (requested_width > 0)
      caps << ",width=" << requested_width;

    if (requested_height > 0)
      caps << ",height=" << requested_height;

    if (requested_fps > 0)
      caps << ",framerate=" << requested_fps << "/1";

    return caps.str();
  }

  std::vector<std::string> MakeCameraPipelines(int device_index, int requested_width, int requested_height, int requested_fps)
  {
    const std::string source = "v4l2src device=/dev/video" + std::to_string(device_index) + " ! ";
    const std::string sink = "videoconvert ! video/x-raw,format=BGR ! appsink name=camera_sink max-buffers=1 drop=true sync=false";

    std::vector<std::string> pipelines;
    if (requested_width > 0 || requested_height > 0 || requested_fps > 0)
    {
      pipelines.push_back(source + MakeCaps("video/x-raw", requested_width, requested_height, requested_fps) + " ! " + sink);
      pipelines.push_back(source + MakeCaps("image/jpeg", requested_width, requested_height, requested_fps) + " ! jpegdec ! " + sink);
    }
    else
    {
      pipelines.push_back(source + sink);
      pipelines.push_back(source + "image/jpeg ! jpegdec ! " + sink);
    }

    return pipelines;
  }

  bool ParseFrameCaps(GstSample* sample, int& width, int& height)
  {
    GstCaps* caps = gst_sample_get_caps(sample);
    if (!caps)
      return false;

    GstStructure* structure = gst_caps_get_structure(caps, 0);
    if (!structure)
      return false;

    return gst_structure_get_int(structure, "width", &width) && gst_structure_get_int(structure, "height", &height);
  }

  bool ParseVideoInfo(GstSample* sample, GstVideoInfo& info)
  {
    GstCaps* caps = gst_sample_get_caps(sample);
    return caps && gst_video_info_from_caps(&info, caps);
  }
} // namespace

namespace camera
{
  struct CameraHandler::Impl
  {
    Impl()
      : frames(2)
    {
    }

    void Run();

    utils::Channel<CameraFrame> frames;
    std::unique_ptr<GstElement, GstElementDeleter> pipeline;
    GstAppSink* sink = nullptr;
    std::atomic<bool> running{false};
    bool opened = false;
    std::thread worker;
    int width = 0;
    int height = 0;
  };

  CameraHandler::CameraHandler(int device_index, int requested_width, int requested_height, int requested_fps)
    : impl_(std::make_unique<Impl>())
  {
    static std::once_flag gst_init_once;
    std::call_once(gst_init_once, []() {
      gst_init(nullptr, nullptr);
    });

    const std::vector<std::string> pipeline_descriptions = MakeCameraPipelines(device_index, requested_width, requested_height, requested_fps);

    for (const std::string& pipeline_description : pipeline_descriptions)
    {
      GError* error = nullptr;
      GstElement* pipeline = gst_parse_launch(pipeline_description.c_str(), &error);
      if (!pipeline)
      {
        std::cerr << "Could not create GStreamer camera pipeline";
        if (error)
        {
          std::cerr << ": " << error->message;
          g_error_free(error);
        }
        std::cerr << "\nPipeline: " << pipeline_description << "\n";
        continue;
      }

      if (error)
      {
        g_error_free(error);
      }

      impl_->pipeline.reset(pipeline);

      GstElement* sink = gst_bin_get_by_name(GST_BIN(impl_->pipeline.get()), "camera_sink");
      if (!sink)
      {
        std::cerr << "Could not find GStreamer appsink in camera pipeline\n";
        impl_->pipeline.reset();
        continue;
      }

      impl_->sink = GST_APP_SINK(sink);
      gst_app_sink_set_emit_signals(impl_->sink, false);
      gst_object_unref(sink);

      const GstStateChangeReturn state = gst_element_set_state(impl_->pipeline.get(), GST_STATE_PLAYING);
      if (state == GST_STATE_CHANGE_FAILURE)
      {
        std::cerr << "Could not start GStreamer camera pipeline\nPipeline: " << pipeline_description << "\n";
        impl_->sink = nullptr;
        impl_->pipeline.reset();
        continue;
      }

      GstSample* sample = gst_app_sink_try_pull_sample(impl_->sink, GST_SECOND);
      if (!sample)
      {
        std::cerr << "Could not read initial camera frame from GStreamer pipeline\nPipeline: " << pipeline_description << "\n";
        impl_->sink = nullptr;
        impl_->pipeline.reset();
        continue;
      }

      if (!ParseFrameCaps(sample, impl_->width, impl_->height))
      {
        std::cerr << "Could not read camera frame dimensions from GStreamer pipeline\nPipeline: " << pipeline_description << "\n";
        gst_sample_unref(sample);
        impl_->sink = nullptr;
        impl_->pipeline.reset();
        continue;
      }

      gst_sample_unref(sample);
      break;
    }

    if (!impl_->pipeline || !impl_->sink)
    {
      std::cerr << "Could not open camera device " << device_index << " with GStreamer\n";
      return;
    }

    impl_->opened = true;
    impl_->running = true;

    std::cerr << "Camera opened with GStreamer. WxH=" << impl_->width << "x" << impl_->height << "\n";
    impl_->worker = std::thread(&Impl::Run, impl_.get());
  }

  CameraHandler::~CameraHandler()
  {
    Stop();
  }

  bool CameraHandler::IsOpened() const
  {
    return impl_ && impl_->opened;
  }

  int CameraHandler::Width() const
  {
    return impl_ ? impl_->width : 0;
  }

  int CameraHandler::Height() const
  {
    return impl_ ? impl_->height : 0;
  }

  utils::Channel<CameraFrame>& CameraHandler::Frames()
  {
    return impl_->frames;
  }

  void CameraHandler::Stop()
  {
    if (impl_)
    {
      impl_->running = false;
      impl_->opened = false;

      if (impl_->pipeline)
        gst_element_set_state(impl_->pipeline.get(), GST_STATE_NULL);
    }

    if (impl_)
      impl_->frames.Close();

    if (impl_ && impl_->worker.joinable())
      impl_->worker.join();
  }

  void CameraHandler::Impl::Run()
  {
    std::uint64_t frame_index = 0;

    while (running)
    {
      GstSample* sample = gst_app_sink_try_pull_sample(sink, kPullTimeout);
      if (!sample)
        continue;

      GstBuffer* buffer = gst_sample_get_buffer(sample);
      if (!buffer)
      {
        gst_sample_unref(sample);
        continue;
      }

      GstVideoInfo video_info;
      if (!ParseVideoInfo(sample, video_info))
      {
        gst_sample_unref(sample);
        continue;
      }

      const int width = GST_VIDEO_INFO_WIDTH(&video_info);
      const int height = GST_VIDEO_INFO_HEIGHT(&video_info);
      const int row_bytes = width * 3;
      const int stride = GST_VIDEO_INFO_PLANE_STRIDE(&video_info, 0);

      GstMapInfo map;
      if (!gst_buffer_map(buffer, &map, GST_MAP_READ))
      {
        gst_sample_unref(sample);
        continue;
      }

      CameraFrame out;
      out.index = frame_index++;
      out.bgr.width = width;
      out.bgr.height = height;
      out.bgr.channels = 3;
      out.bgr.pixels.resize(static_cast<size_t>(row_bytes * height));

      if (stride == row_bytes)
      {
        std::memcpy(out.bgr.pixels.data(), map.data, out.bgr.pixels.size());
      }
      else
      {
        for (int y = 0; y < height; ++y)
        {
          std::memcpy(
            out.bgr.pixels.data() + static_cast<size_t>(y * row_bytes),
            map.data + static_cast<size_t>(y * stride),
            static_cast<size_t>(row_bytes));
        }
      }

      gst_buffer_unmap(buffer, &map);
      gst_sample_unref(sample);

      if (!frames.Send(std::move(out)))
        break;
    }

    frames.Close();
  }
} // namespace camera
