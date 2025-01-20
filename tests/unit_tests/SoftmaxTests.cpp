#include <snn/neural_network/optimizer/Softmax.hpp>

#include "../ExtendedGTest.hpp"
using namespace std;
using namespace snn::internal;

TEST(Softmax, NormalValues)
{
    vector<float> values1{7.0f, 2.0f, 9.0f};
    vector<float> values2 = values1;
    const vector<float> expectedValues{0.119107f, 0.000803f, 0.880090f};
    Softmax softmax(nullptr);

    softmax.applyAfterOutputForTraining(values1, false);
    softmax.applyAfterOutputForTesting(values2);

    ASSERT_VECTOR_EQ(values1, expectedValues);
    ASSERT_VECTOR_EQ(values2, expectedValues);
}

TEST(Softmax, LargeValues)
{
    vector<float> values1{15012.0f, 15009.0f, 15011.0f};
    vector<float> values2 = values1;
    const vector<float> expectedValues{0.705384f, 0.035119f, 0.259496f};
    Softmax softmax(nullptr);

    softmax.applyAfterOutputForTraining(values1, false);
    softmax.applyAfterOutputForTesting(values2);

    ASSERT_VECTOR_EQ(values1, expectedValues);
    ASSERT_VECTOR_EQ(values2, expectedValues);
}

TEST(Softmax, SmallValues)
{
    vector<float> values1{7e-5f, -3e-6f, 1e-6f};
    vector<float> values2 = values1;
    const vector<float> expectedValues{0.333349f, 0.333324f, 0.333326f};
    Softmax softmax(nullptr);

    softmax.applyAfterOutputForTraining(values1, false);
    softmax.applyAfterOutputForTesting(values2);

    ASSERT_VECTOR_EQ(values1, expectedValues);
    ASSERT_VECTOR_EQ(values2, expectedValues);
}
