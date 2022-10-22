#pragma once
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

namespace snn::internal
{
    class Tensor final : public std::vector<float>
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        using vector<float>::vector;

    };

    template <class Archive>
    void Tensor::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        boost::serialization::void_cast_register<Tensor, std::vector<float>>();
        ar& boost::serialization::base_object<std::vector<float>>(*this);
    }
}
