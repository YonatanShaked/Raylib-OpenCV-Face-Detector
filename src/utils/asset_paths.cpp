#include "utils/asset_paths.h"

namespace utils
{
  std::filesystem::path AssetPath(const std::filesystem::path& rel)
  {
    return std::filesystem::current_path() / "assets" / rel;
  }
} // namespace utils
