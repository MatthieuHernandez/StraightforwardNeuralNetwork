#pragma once
#include <algorithm>
#include <vector>

#include "Circular.hpp"

namespace snn::internal
{
class InputCircular final : public Circular<std::vector<float>>
{
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& archive, uint32_t version);

    public:
        template <std::same_as<float>... Values>
        void pushBack(const std::vector<float>& data, [[maybe_unused]] Values... extraValues)
        {
            this->indexPush++;
            if (this->indexPush >= this->queue.size())
            {
                this->indexPush = 0;
            }
            auto size = data.size();
            auto& inputs = this->queue[this->indexPush];
            inputs.resize(size + sizeof...(extraValues));
            std::ranges::copy(data, inputs.begin());
            ((inputs[size++] = extraValues), ...);
        }
};

template <class Archive>
void InputCircular::serialize(Archive& archive, [[maybe_unused]] const uint32_t version)
{
    boost::serialization::void_cast_register<InputCircular, Circular<std::vector<float>>>();
    archive& boost::serialization::base_object<Circular<std::vector<float>>>(*this);
}
}  // namespace snn::internal
