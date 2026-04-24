#ifndef UTILS_ASSET_PATHS_H
#define UTILS_ASSET_PATHS_H

#include <filesystem>

namespace utils
{
  std::filesystem::path AssetPath(const std::filesystem::path& rel);
} // namespace utils

#endif // UTILS_ASSET_PATHS_H
