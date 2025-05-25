#include "Circular.hpp"

#include <cassert>

namespace snn::internal
{
template class Circular<float>;
template class Circular<std::vector<float>>;

template <>
void Circular<float>::initialize(const size_t size, [[maybe_unused]] const size_t dataSize)
{
    assert(dataSize == 1);
    this->queue.clear();
    this->queue.resize(size);
}
template <>
void Circular<std::vector<float>>::initialize(const size_t size, const size_t dataSize)
{
    this->queue.clear();
    this->queue.resize(size);
    for (auto& d : this->queue)
    {
        d = std::vector<float>(dataSize, 0.0F);
    }
}
}  // namespace snn::internal