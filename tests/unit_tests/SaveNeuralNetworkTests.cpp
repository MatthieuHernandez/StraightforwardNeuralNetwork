#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/neural_network/layer/neuron/Neuron.hpp>

#include "../ExtendedGTest.hpp"

using namespace std;
using namespace snn;

TEST(SaveNeuralNetwork, EqualTest)
{
    auto structureOfNetwork = {Input(3, 8, 8),
                               LocallyConnected(2, 2),
                               Convolution(2, 4, activation::ReLU),
                               FullyConnected(20, activation::iSigmoid, L1Regularization(1e-3f)),
                               LocallyConnected(3, 2, activation::tanh, L2Regularization(1e-3f)),
                               FullyConnected(5, activation::sigmoid, Dropout(0.0f)),
                               Convolution(1, 1, activation::tanh),
                               GruLayer(3),
                               Recurrence(4)};
    StraightforwardNeuralNetwork A(structureOfNetwork, StochasticGradientDescent(0.9f, 0.78f));
    StraightforwardNeuralNetwork C(structureOfNetwork, StochasticGradientDescent(0.9f, 0.78f));
    StraightforwardNeuralNetwork B = A;

    ASSERT_EQ(A.isValid(), 0);
    ASSERT_EQ(B.isValid(), 0);

    EXPECT_TRUE(A == B);
    EXPECT_TRUE(&A != &B);

    EXPECT_TRUE(&A != &B);

    EXPECT_TRUE(*A.optimizer == *B.optimizer);
    EXPECT_TRUE(A.optimizer != B.optimizer);

    EXPECT_TRUE(*A.layers[0] == *B.layers[0]);
    EXPECT_TRUE(A.layers[0] != B.layers[0]);

    auto* neuronA = static_cast<internal::SimpleNeuron*>(A.layers[0]->getNeuron(0));
    auto* neuronB = static_cast<internal::SimpleNeuron*>(B.layers[0]->getNeuron(0));
    EXPECT_TRUE(*neuronA == *neuronB);
    EXPECT_TRUE(neuronA != neuronB);

    EXPECT_TRUE(A.optimizer.get() == static_cast<internal::SimpleNeuron*>(A.layers[0]->getNeuron(0))->getOptimizer());

    EXPECT_TRUE(A != C);  // Test A == C with same seed

    vector<float> inputs(8 * 8 * 3);
    inputs[29] = 0.99f;
    inputs[30] = 0.88f;
    inputs[60] = 0.75f;
    inputs[90] = -0.25f;
    inputs[120] = 1.0f;
    inputs[150] = -0.77f;
    const vector<float> desired{1.0f, 0.0f, 0.5f, 0.07f};

    for (int i = 0; i < 10; ++i) A.trainOnce(inputs, desired);

    EXPECT_TRUE(A != B);

    for (int i = 0; i < 10; ++i) B.trainOnce(inputs, desired);

    EXPECT_TRUE(A == B);

    EXPECT_TRUE(A.getF1Score() == B.getF1Score()) << "A == B";
    EXPECT_TRUE(A.getGlobalClusteringRate() == B.getGlobalClusteringRate()) << "A == B";
    EXPECT_TRUE(A.getWeightedClusteringRate() == B.getWeightedClusteringRate()) << "A == B";

    // A.~StraightforwardNeuralNetwork(); // doesn't work on Linux
    // B.trainOnce(inputs, desired);
}

TEST(SaveNeuralNetwork, EqualTestWithDropout)
{
    auto structureOfNetwork = {Input(10), FullyConnected(200, activation::sigmoid, Dropout(0.4f)), FullyConnected(4)};
    StraightforwardNeuralNetwork A(structureOfNetwork);
    StraightforwardNeuralNetwork B = A;

    vector<float> inputs(10);
    inputs[1] = 1.5f;
    inputs[4] = 0.75f;
    inputs[5] = -0.25f;
    inputs[7] = 1.0f;
    inputs[9] = -1.35f;
    const vector<float> desired{1.0f, 0.0f, 0.5f, 0.07f};

    EXPECT_TRUE(A == B);

    A.trainOnce(inputs, desired);
    B.trainOnce(inputs, desired);

    EXPECT_TRUE(A != B);

    EXPECT_TRUE(A.getF1Score() == B.getF1Score()) << "A == B";
    EXPECT_TRUE(A.getGlobalClusteringRate() == B.getGlobalClusteringRate()) << "A == B";
    EXPECT_TRUE(A.getWeightedClusteringRate() == B.getWeightedClusteringRate()) << "A == B";
}

TEST(SaveNeuralNetwork, Save)  // TODO: do a forward to be sure that the network is the same.
{
    StraightforwardNeuralNetwork A(
        {Input(45), MaxPooling(3), Convolution(2, 2, activation::ReLU),
         LocallyConnected(2, 2, activation::tanh, L1Regularization(1e-5f)),
         FullyConnected(2, activation::sigmoid, Dropout(0.0f)), GruLayer(2, L2Regularization(1e-5f))},
        StochasticGradientDescent(0.03f, 0.78f));

    auto randomInput = tools::randomVector(0, 2, 45);
    auto randomOutput = tools::randomVector(0, 1, 2);

    A.saveAs("./testSave.tmp");
    A.trainOnce(randomInput, randomOutput);
    auto outputA = A.computeOutput(randomInput);

    StraightforwardNeuralNetwork B = StraightforwardNeuralNetwork::loadFrom("./testSave.tmp");

    B.trainOnce(randomInput, randomOutput);
    auto outputB = B.computeOutput(randomInput);

    EXPECT_TRUE(A == B);
    EXPECT_TRUE(sizeof(A) == sizeof(B));  // Don't count pointing objects by pointer
    EXPECT_TRUE(outputA == outputB);
    ASSERT_EQ(B.isValid(), 0);
}
