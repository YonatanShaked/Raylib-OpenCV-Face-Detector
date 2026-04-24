#ifndef CHANNEL_H
#define CHANNEL_H

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace utils
{
  template <typename T> class Channel
  {
  public:
    explicit Channel(std::size_t capacity)
      : capacity_(capacity)
      , closed_(false)
    {
    }

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    bool Send(const T& value)
    {
      std::unique_lock<std::mutex> lock(mutex_);

      send_cv_.wait(lock,
                    [this]()
                    {
                      return closed_ || queue_.size() < capacity_;
                    });

      if (closed_)
        return false;

      queue_.push_back(value);
      recv_cv_.notify_one();
      return true;
    }

    bool Send(T&& value)
    {
      std::unique_lock<std::mutex> lock(mutex_);

      send_cv_.wait(lock,
                    [this]()
                    {
                      return closed_ || queue_.size() < capacity_;
                    });

      if (closed_)
        return false;

      queue_.push_back(std::move(value));
      recv_cv_.notify_one();
      return true;
    }

    bool Recv(T& value)
    {
      std::unique_lock<std::mutex> lock(mutex_);

      recv_cv_.wait(lock,
                    [this]()
                    {
                      return closed_ || !queue_.empty();
                    });

      if (queue_.empty())
        return false;

      value = std::move(queue_.front());
      queue_.pop_front();
      send_cv_.notify_one();
      return true;
    }

    bool TryRecv(T& value)
    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (queue_.empty())
        return false;

      value = std::move(queue_.front());
      queue_.pop_front();
      send_cv_.notify_one();
      return true;
    }

    void Close()
    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (closed_)
        return;

      closed_ = true;
      recv_cv_.notify_all();
      send_cv_.notify_all();
    }

  private:
    std::size_t capacity_;
    bool closed_;
    std::deque<T> queue_;
    std::mutex mutex_;
    std::condition_variable send_cv_;
    std::condition_variable recv_cv_;
  };
} // namespace utils

#endif
