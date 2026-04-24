#ifndef UTILS_ASSET_PATHS_H
#define UTILS_ASSET_PATHS_H

#include <filesystem>

namespace utils
{
  inline std::filesystem::path AssetPath(const std::filesystem::path& rel)
  {
    return std::filesystem::current_path() / "assets" / rel;
  }
} // namespace utils

#endif // UTILS_ASSET_PATHS_H
