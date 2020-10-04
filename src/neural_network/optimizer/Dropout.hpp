#include <vector>
#include <boost/serialization/access.hpp>
#include "LayerOptimizer.hpp"

namespace snn::internal
{
    class Dropout final : public LayerOptimizer
    {
    private:
        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

        float reverseValue;

    public:
        const float value = 0.1f;

        Dropout(float value);
        Dropout(const Dropout& sgd) = default;
        ~Dropout() = default;

        void apply(std::vector<float>& output) override;
        void applyForBackpropagation(std::vector<float>& output) override;

        bool operator==(const Dropout& d) const;

        bool operator!=(const Dropout& d) const;
    };

    template <class Archive>
    void Dropout::serialize(Archive& ar, const unsigned int version)
    {
        ar & this->value;
        ar & this->reverseValue;
    }
}
