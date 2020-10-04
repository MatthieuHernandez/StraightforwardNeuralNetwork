#pragma once

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

        virtual bool operator==(const Optimizer& optimizer) const
        {
            return typeid(*this).hash_code() == typeid(optimizer).hash_code();
        }

        virtual bool operator!=(const Optimizer& optimizer) const
        {
            return !(*this == optimizer);
        }
    };

    template <class Archive>
    void Optimizer::serialize(Archive& ar, unsigned version)
    {
    }
}
