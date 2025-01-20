#include "Tools.hpp"

#include <algorithm>

namespace snn::tools
{
int randomBetween(const int min, const int max)  // [min; max[
{
    std::uniform_real_distribution<> dist(min, max);
    return static_cast<int>(dist(rng));
}

float randomBetween(const float min, const float max)
{
    std::uniform_real_distribution<> dist(min, max);
    return static_cast<float>(dist(rng));
}

std::vector<float> randomVector(const float min, const float max, const size_t size)
{
    std::uniform_real_distribution<> dist(min, max);
    std::vector<float> vector(size);
    std::ranges::generate(vector, [&] { return static_cast<float>(dist(rng)); });
    return vector;
}
auto toString(std::chrono::milliseconds duration) -> std::string
{
    std::string str;

    if (duration.count() > 3600000)
        return std::to_string(duration.count() / 3600000) + "h " + std::to_string(duration.count() / 60000) + "min";

    if (duration.count() > 60000)
        return std::to_string(duration.count() / 60000) + "min " + std::to_string(duration.count() / 1000) + "s";

    if (duration.count() > 1000) return std::to_string(duration.count() / 1000) + "s";

    if (duration.count() > 0) return std::to_string(duration.count()) + "ms";

    return "";
}
auto toString(const ErrorType err) -> std::string { return std::to_string(static_cast<uint8_t>(err)); }
}  // namespace snn::tools