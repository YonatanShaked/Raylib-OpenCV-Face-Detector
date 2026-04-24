#ifndef ASSETS_PATHS_H
#define ASSETS_PATHS_H

#include <filesystem>

namespace assets
{
  std::filesystem::path AssetPath(const std::filesystem::path& rel);
} // namespace assets

#endif // ASSETS_PATHS_H
