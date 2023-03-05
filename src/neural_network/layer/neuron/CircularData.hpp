#pragma once
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

namespace snn::internal
{
    class CircularData final
    {
    private:
        friend class StochasticGradientDescent;
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        std::vector<std::vector<float>> queue;
        size_t indexPush = 0;
        size_t indexGet = 0;

    public:
        CircularData() = default; // use restricted to Boost library only
        void initialize(size_t queueSize, size_t dataSize); // should be call after the ctor
        ~CircularData() = default;

        [[nodiscard]] std::vector<float> getBack();
        void pushBack(std::vector<float> data);

        bool operator==(const CircularData& other) const;
        bool operator!=(const CircularData& other) const;
    };

    template <class Archive>
    void CircularData::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        ar & queue;
        ar & indexGet;
        ar & indexPush;
    }
}
