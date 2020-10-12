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

        float reverseValue{};

    public:
        const float value = 0.1f;
        Dropout() = default;  // use restricted to Boost library only
        Dropout(float value);
        Dropout(const Dropout& dropout) = default;
        ~Dropout() = default;

        std::unique_ptr<LayerOptimizer> clone(LayerOptimizer* optimizer) const override;

        void apply(std::vector<float>& output) override;
        void applyForBackpropagation(std::vector<float>& output) override;

        bool operator==(const Optimizer& optimizer) const override;
        bool operator!=(const Optimizer& optimizer) const override;
    };

    template <class Archive>
    void Dropout::serialize(Archive& ar, const unsigned int version)
    {
        boost::serialization::void_cast_register<Dropout, LayerOptimizer>();
        ar & boost::serialization::base_object<LayerOptimizer>(*this);
        ar & this->value;
        ar & this->reverseValue;
    }
}
