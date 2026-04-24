#include "asset_utils.h"

namespace assets
{
  std::filesystem::path AssetPath(const std::filesystem::path& rel)
  {
    return std::filesystem::current_path() / "assets" / rel;
  }
} // namespace assets
