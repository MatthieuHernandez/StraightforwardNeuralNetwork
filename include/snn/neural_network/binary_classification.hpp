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
            archive & truePositive;
            archive & trueNegative;
            archive & falsePositive;
            archive & falseNegative;
        }
} __attribute__((packed, aligned(32)));  // NOLINT(*magic-numbers)
}  // namespace snn::internal