#include <snn/neural_network/optimizer/Softmax.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn::internal;

TEST(Softmax, NormalValues)
{
    std::vector<float> values1{7.0F, 2.0F, 9.0F};
    std::vector<float> values2 = values1;
    const std::vector<float> expectedValues{0.119107F, 0.000803F, 0.880090F};
    Softmax softmax(nullptr);

    softmax.applyAfterOutputForTraining(values1, false);
    softmax.applyAfterOutputForTesting(values2);

    ASSERT_VECTOR_EQ(values1, expectedValues);
    ASSERT_VECTOR_EQ(values2, expectedValues);
}

TEST(Softmax, LargeValues)
{
    std::vector<float> values1{15012.0F, 15009.0F, 15011.0F};
    std::vector<float> values2 = values1;
    const std::vector<float> expectedValues{0.705384F, 0.035119F, 0.259496F};
    Softmax softmax(nullptr);

    softmax.applyAfterOutputForTraining(values1, false);
    softmax.applyAfterOutputForTesting(values2);

    ASSERT_VECTOR_EQ(values1, expectedValues);
    ASSERT_VECTOR_EQ(values2, expectedValues);
}

TEST(Softmax, SmallValues)
{
    std::vector<float> values1{7e-5F, -3e-6F, 1e-6F};
    std::vector<float> values2 = values1;
    const std::vector<float> expectedValues{0.333349F, 0.333324F, 0.333326F};
    Softmax softmax(nullptr);

    softmax.applyAfterOutputForTraining(values1, false);
    softmax.applyAfterOutputForTesting(values2);

    ASSERT_VECTOR_EQ(values1, expectedValues);
    ASSERT_VECTOR_EQ(values2, expectedValues);
}
