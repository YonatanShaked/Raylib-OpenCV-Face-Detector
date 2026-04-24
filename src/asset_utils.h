#ifndef ASSET_UTILS_H
#define ASSET_UTILS_H

#include <filesystem>

namespace assets
{
  std::filesystem::path AssetPath(const std::filesystem::path& rel);
} // namespace assets

#endif // ASSET_UTILS_H
