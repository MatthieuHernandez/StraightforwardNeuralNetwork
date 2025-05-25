#pragma once

#include <cassert>

#include "Circular.hpp"

namespace snn::internal
{
template <typename T>
auto Circular<T>::getBack() -> const T*
{
    assert(this->indexGet <= this->queue.size());
    if (this->indexGet >= this->queue.size())
    {
        this->indexGet = 0;
    }
    return &this->queue[this->indexGet++];
}

template <typename T>
void Circular<T>::pushBack(const T& data)
{
    assert(this->indexPush <= this->queue.size());
    if (this->indexPush >= this->queue.size())
    {
        this->indexPush = 0;
    }
    this->queue[this->indexPush++] = data;
}
}  // namespace snn::internal