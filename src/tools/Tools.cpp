#include "Tools.hpp"

#include <algorithm>

namespace snn::tools
{
auto Rng() -> std::mt19937&
{
    static constexpr uint32_t seed = 42;
    static std::mt19937 rng = useFixedSeed ? std::mt19937{seed} : std::mt19937{std::random_device{}()};
    return rng;
}

auto randomBetween(const int min, const int max) -> int  // [min; max[
{
    std::uniform_real_distribution<> dist(min, max);
    return static_cast<int>(dist(Rng()));
}

auto randomBetween(const float min, const float max) -> float
{
    std::uniform_real_distribution<> dist(min, max);
    return static_cast<float>(dist(Rng()));
}

auto randomVector(const float min, const float max, const size_t size) -> std::vector<float>
{
    std::uniform_real_distribution<> dist(min, max);
    std::vector<float> vector(size);
    std::ranges::generate(vector, [&] { return static_cast<float>(dist(tools::Rng())); });
    return vector;
}

auto toString(std::chrono::milliseconds duration) -> std::string
{
    const std::string str;

    if (duration.count() > 3600000)
    {
        return std::to_string(duration.count() / 3600000) + "h " + std::to_string(duration.count() / 60000) + "min";
    }

    if (duration.count() > 60000)
    {
        return std::to_string(duration.count() / 60000) + "min " + std::to_string(duration.count() / 1000) + "s";
    }

    if (duration.count() > 1000)
    {
        return std::to_string(duration.count() / 1000) + "s";
    }

    if (duration.count() > 0)
    {
        return std::to_string(duration.count()) + "ms";
    }

    return "";
}

auto toString(const errorType err) -> std::string { return std::to_string(static_cast<uint8_t>(err)); }
}  // namespace snn::tools
