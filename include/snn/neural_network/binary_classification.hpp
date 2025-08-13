#pragma once
#include <boost/serialization/access.hpp>
#include <cstdint>

namespace snn::internal
{
struct binaryClassification
{
        float truePositive{};
        float trueNegative{};
        float falsePositive{};
        float falseNegative{};
        float totalError{};

        auto operator==(const binaryClassification&) const -> bool = default;

        template <typename Archive>
        void serialize(Archive& archive, [[maybe_unused]] uint32_t version)
        {
            archive& this->truePositive;
            archive& this->trueNegative;
            archive& this->falsePositive;
            archive& this->falseNegative;
            archive& this->totalError;
        }
};
}  // namespace snn::internal