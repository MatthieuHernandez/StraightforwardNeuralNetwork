#pragma once
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <vector>

#include "../../../tools/Tools.hpp"

namespace snn::internal
{
class CircularData final
{
    private:
        friend class StochasticGradientDescent;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        vector2D<float> queue;
        size_t indexPush = 0;
        size_t indexGet = 0;

    public:
        CircularData() = default;                            // use restricted to Boost library only
        void initialize(size_t queueSize, size_t dataSize);  // should be call after the ctor
        ~CircularData() = default;

        [[nodiscard]] auto getBack() -> const std::vector<float>*;
        void pushBack(const std::vector<float>& data);

        auto operator==(const CircularData& other) const -> bool;
        auto operator!=(const CircularData& other) const -> bool;
};

template <class Archive>
void CircularData::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
{
    ar & queue;
    ar & indexGet;
    ar & indexPush;
}
}  // namespace snn::internal
