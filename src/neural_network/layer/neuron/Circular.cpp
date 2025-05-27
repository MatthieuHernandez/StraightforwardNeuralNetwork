#include "Circular.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace snn::internal
{
template class Circular<float>;
template class Circular<std::vector<float>>;

template <>
void Circular<float>::initialize(const size_t size, [[maybe_unused]] const size_t dataSize)
{
    assert(dataSize == 1);
    this->divider = static_cast<float>(size);
    this->queue.clear();
    this->queue.resize(size);
}
template <>
void Circular<std::vector<float>>::initialize(const size_t size, const size_t dataSize)
{
    this->divider = static_cast<float>(size);
    this->queue.clear();
    this->queue.resize(size);
    for (auto& d : this->queue)
    {
        d = std::vector<float>(dataSize, 0.0F);
    }
}

template <>
auto Circular<std::vector<float>>::getSum() -> std::vector<float>
{
    const std::size_t dataSize = this->queue.front().size();
    std::vector<float> result(dataSize, 0.0F);
    for (const auto& data : this->queue)
    {
        assert(data.size() == dataSize);
        std::transform(data.cbegin(), data.cend(), result.begin(), result.begin(), std::plus<>{});
    }
    return result;
}

template <>
auto Circular<float>::getSum() -> float
{
    auto result = std::reduce(this->queue.cbegin(), this->queue.cend());
    return result;
}

template <>
auto Circular<float>::getAverage() -> float
{
    auto result = std::reduce(this->queue.cbegin(), this->queue.cend());
    result /= this->divider;
    return result;
}

template <>
auto Circular<std::vector<float>>::getAverage() -> std::vector<float>
{
    const std::size_t dataSize = this->queue.front().size();
    std::vector<float> result(dataSize, 0.0F);
    for (const auto& data : this->queue)
    {
        assert(data.size() == dataSize);
        std::transform(data.cbegin(), data.cend(), result.begin(), result.begin(), std::plus<>{});
        std::transform(result.begin(), result.end(), result.begin(), [&](float x) { return x / this->divider; });
    }
    return result;
}

template <>
auto Circular<std::vector<float>>::MultiplyAndAccumulate(const Circular<float>& multiplier) -> std::vector<float>
{
    const std::size_t dataSize = this->queue.front().size();
    std::vector<float> result(dataSize, 0.0F);

    assert(this->queue.size() == multiplier.queue.size());
    for (std::size_t q = 0; q < this->queue.size(); ++q)
    {
        const auto& data = this->queue[q];
        const float multi = multiplier.queue[q];

        assert(data.size() == dataSize);
        for (std::size_t d = 0; d < dataSize; ++d)
        {
            result[d] += data[d] * multi;
        }
    }
    return result;
}
}  // namespace snn::internal
