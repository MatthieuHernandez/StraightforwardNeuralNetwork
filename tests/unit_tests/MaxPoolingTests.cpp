#include <boost/serialization/smart_cast.hpp>
#include <memory>
#include <numeric>
#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/tools/Tools.hpp>

#include "../ExtendedGTest.hpp"

using namespace std;
using namespace snn;

TEST(MaxPooling, SimpleMaxPooling1D)
{
    vector<float> input(9);
    vector<float> error(6);
    std::iota(std::begin(input), std::end(input), 1.0F);
    std::iota(std::begin(error), std::end(error), 1.0F);

    vector<float> expectedOutput{4, 5, 6, 7, 8, 9};
    vector<float> expectedBackOutput{0, 0, 0, 1, 2, 3, 4, 5, 6};
    LayerModel model{layerType::maxPooling, 9, 0, 6, {-1, -1, -1, 0.0F, activation::identity}, 3, 6, 2, 2, {3, 3}, {}};

    internal::MaxPooling1D pooling(model);
    auto output = pooling.output(input, false);
    auto backOutput = pooling.backOutput(error);

    ASSERT_EQ(output, expectedOutput);
    ASSERT_EQ(backOutput, expectedBackOutput);
}

TEST(MaxPooling, SimpleMaxPooling2D)
{
    vector<float> input{3, 1, 4, 2, 1, 1, 1, 1, 6, 2, 3, 1, 3, 1, 1, 2, 5, 1, 1, 1, 7, 1, 1, 1, 7,
                        2, 4, 1, 4, 1, 1, 2, 2, 1, 6, 1, 5, 1, 8, 2, 9, 2, 2, 1, 2, 2, 1, 1, 1, 2};
    vector<float> error(18);
    std::iota(std::begin(error), std::end(error), 1.0F);

    vector<float> expectedOutput{4, 2, 5, 2, 6, 2, 7, 2, 7, 2, 8, 2, 9, 2, 2, 2, 1, 2};
    vector<float> expectedBackOutput{0, 0, 1, 2, 0,  0,  0,  0,  5,  6, 0,  0,  0, 0, 0,  4, 3,
                                     0, 0, 0, 7, 0,  0,  0,  9,  10, 0, 0,  0,  0, 0, 8,  0, 0,
                                     0, 0, 0, 0, 11, 12, 13, 14, 0,  0, 15, 16, 0, 0, 17, 18};
    LayerModel model{
        layerType::maxPooling, 50, 0, 18, {-1, -1, -1, 0.0F, activation::identity}, 2, 18, 9, 2, {2, 5, 5}, {}};

    internal::MaxPooling2D pooling(model);
    auto output = pooling.output(input, false);
    auto backOutput = pooling.backOutput(error);

    ASSERT_EQ(output, expectedOutput);
    ASSERT_EQ(backOutput, expectedBackOutput);
}
