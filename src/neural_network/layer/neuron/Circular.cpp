#include "Circular.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace snn::internal
{
template class Circular<float>;
template class Circular<std::vector<float>>;

template <>
void Circular<float>::initialize(const size_t queueSize, [[maybe_unused]] const size_t dataSize)
{
    assert(dataSize == 1);
    this->indexPush = 0;
    this->indexGet = 0;
    this->divider = static_cast<float>(queueSize);
    this->queue.assign(queueSize, 0.0F);
}
template <>
void Circular<std::vector<float>>::initialize(const size_t queueSize, const size_t dataSize)
{
    this->indexPush = 0;
    this->indexGet = 0;
    this->divider = static_cast<float>(queueSize);
    this->queue.resize(queueSize);
    for (auto& d : this->queue)
    {
        d = std::vector<float>(dataSize, 0.0F);
    }
}

template <>
void Circular<float>::reset()
{
    this->indexPush = 0;
    this->indexGet = 0;
    std::ranges::fill(this->queue, 0.0F);
}

template <>
void Circular<std::vector<float>>::reset()
{
    this->indexPush = 0;
    this->indexGet = 0;
    for (auto& d : this->queue)
    {
        std::ranges::fill(d, 0.0F);
    }
}

template <>
auto Circular<std::vector<float>>::getSum() const -> std::vector<float>
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
auto Circular<float>::getSum() const -> float
{
    auto result = std::reduce(this->queue.cbegin(), this->queue.cend());
    return result;
}

template <>
auto Circular<float>::getAverage() const -> float
{
    auto result = std::reduce(this->queue.cbegin(), this->queue.cend());
    result /= this->divider;
    return result;
}

template <>
auto Circular<std::vector<float>>::getAverage() const -> std::vector<float>
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
auto Circular<std::vector<float>>::MultiplyAndAccumulate(const Circular<float>& multiplier) const -> std::vector<float>
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
