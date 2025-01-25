#include <snn/neural_network/StraightforwardNeuralNetwork.hpp>
#include <snn/neural_network/layer/neuron/Neuron.hpp>

#include "../ExtendedGTest.hpp"

using namespace snn;

TEST(SaveNeuralNetwork, EqualTest)
{
    auto structureOfNetwork = {Input(3, 8, 8),
                               LocallyConnected(2, 2),
                               Convolution(2, 4, activation::ReLU),
                               FullyConnected(20, activation::iSigmoid, L1Regularization(1e-3F)),
                               LocallyConnected(3, 2, activation::tanh, L2Regularization(1e-3F)),
                               FullyConnected(5, activation::sigmoid, Dropout(0.0F)),
                               Convolution(1, 1, activation::tanh),
                               GruLayer(3),
                               Recurrence(4)};
    StraightforwardNeuralNetwork A(structureOfNetwork, StochasticGradientDescent(0.9F, 0.78F));
    StraightforwardNeuralNetwork C(structureOfNetwork, StochasticGradientDescent(0.9F, 0.78F));
    StraightforwardNeuralNetwork B = A;

    ASSERT_EQ(A.isValid(), ErrorType::noError);
    ASSERT_EQ(B.isValid(), ErrorType::noError);

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

    std::vector<float> inputs(8 * 8 * 3);
    inputs[29] = 0.99F;
    inputs[30] = 0.88F;
    inputs[60] = 0.75F;
    inputs[90] = -0.25F;
    inputs[120] = 1.0F;
    inputs[150] = -0.77F;
    const std::vector<float> desired{1.0F, 0.0F, 0.5F, 0.07F};

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
    auto structureOfNetwork = {Input(10), FullyConnected(200, activation::sigmoid, Dropout(0.4F)), FullyConnected(4)};
    StraightforwardNeuralNetwork A(structureOfNetwork);
    StraightforwardNeuralNetwork B = A;

    std::vector<float> inputs(10);
    inputs[1] = 1.5F;
    inputs[4] = 0.75F;
    inputs[5] = -0.25F;
    inputs[7] = 1.0F;
    inputs[9] = -1.35F;
    const std::vector<float> desired{1.0F, 0.0F, 0.5F, 0.07F};

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
         LocallyConnected(2, 2, activation::tanh, L1Regularization(1e-5F)),
         FullyConnected(2, activation::sigmoid, Dropout(0.0F)), GruLayer(2, L2Regularization(1e-5F))},
        StochasticGradientDescent(0.03F, 0.78F));

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
    ASSERT_EQ(B.isValid(), ErrorType::noError);
}
