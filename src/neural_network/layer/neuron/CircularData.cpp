#include "CircularData.hpp"

#include <cassert>
#include <utility>

using namespace std;
using namespace snn::internal;

void CircularData::initialize(const size_t queueSize, const size_t dataSize)
{
    this->queue.clear();
    this->queue.resize(queueSize);
    for (auto& d : this->queue) d = vector<float>(dataSize, 0.0f);
}

auto CircularData::getBack() -> const vector<float>*
{
    assert(this->indexGet <= this->queue.size());
    if (this->indexGet >= this->queue.size()) this->indexGet = 0;
    return &this->queue[this->indexGet++];
}

void CircularData::pushBack(const vector<float>& data)
{
    assert(this->indexPush <= this->queue.size());
    if (this->indexPush >= this->queue.size()) this->indexPush = 0;
    this->queue[this->indexPush++] = std::move(data);
}

auto CircularData::operator==(const CircularData& other) const -> bool
{
    return this->queue == other.queue && this->indexGet == other.indexGet && this->indexPush == other.indexPush;
}

auto CircularData::operator!=(const CircularData& other) const -> bool { return !(*this == other); }
