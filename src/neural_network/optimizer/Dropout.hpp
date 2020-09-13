#include <boost/serialization/access.hpp>
#include "Optimizer.hpp"

namespace snn::internal
{
    class Dropout : public Optimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    public:
        float value = 0.1f;

        Dropout(float value) : value(value) {}
        Dropout(const Dropout& sgd) = default;
        ~Dropout() = default;

        bool operator==(const Dropout& d) const 
        {
            return this->value == d.value;
        }
        bool operator!=(const Dropout& d) const 
        { 
            return !(*this == d); 
        }
    };

    template <class Archive>
    void Dropout::serialize(Archive& ar, const unsigned int version)
    {
        ar & this->value;
    }
}
