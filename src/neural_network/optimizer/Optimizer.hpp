#pragma once
#include <boost/serialization/access.hpp>

namespace snn::internal
{
    class Optimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        virtual ~Optimizer() = default;

        virtual bool operator==(const Optimizer& optimizer) const;
        virtual bool operator!=(const Optimizer& optimizer) const;
    };

    template <class Archive>
    void Optimizer::serialize(Archive& ar, unsigned version)
    {
    }
}
