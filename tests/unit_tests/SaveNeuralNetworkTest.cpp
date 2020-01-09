#include "../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

TEST(SaveNeuralNetwork, EqualTest)
{
    auto structureOfNetwork = {
        AllToAll(20, iSigmoid),
        AllToAll(10, snn::tanh),
        AllToAll(3, sigmoid)
    };
    StraightforwardNeuralNetwork A(5, structureOfNetwork);
    StraightforwardNeuralNetwork C(5, structureOfNetwork);
    A.learningRate = 0.03f;
    A.momentum = 0.78f;
    C.learningRate = 0.03f;
    C.momentum = 0.78f;
    StraightforwardNeuralNetwork B = A;

    ASSERT_EQ(A.isValid(), 0) << "A is invalid";
    ASSERT_EQ(B.isValid(), 0) << "B is invalid";

    EXPECT_TRUE(A == B);
    EXPECT_TRUE(&A != &B);

    EXPECT_TRUE(&A != &B) << "&A != &B";

    EXPECT_TRUE(*A.layers[0] == *B.layers[0]) << "Value : A.layers[0] == B.layers[0]";
    EXPECT_TRUE(A.layers[0] != B.layers[0]) << "Address : A.layers[0] != B.layers[0]";

    //EXPECT_TRUE(*&A.layers[0].neurons[0] == *&B.layers[0].neurons[0]) << "Value : A.Layers[0].neurons[0] == B.Layers[0].neurons[0]";
    //EXPECT_TRUE(&A.layers[0].neurons[0] != &B.layers[0].neurons[0]) << "Address : A.Layers[0].neurons[0] != B.Layers[0].neurons[0]";
    
    //EXPECT_TRUE(*A.layers[0].neurons[0].getActivationFunction() == *B.layers[0].neurons[0].getActivationFunction()) << "Value : A.Layers[0].neurons[0] == B.Layers[0].neurons[0]";
    //EXPECT_TRUE(A.layers[0].neurons[0].getActivationFunction() != B.layers[0].neurons[0].getActivationFunction()) << "Address : A.Layers[0].neurons[0] != B.Layers[0].neurons[0]";

    EXPECT_TRUE(A != C); // Test A == C with same seed

    const vector<float> inputs {1.5, 0.75, -0.25, 0, 0};
    const vector<float> desired {1, 0, 0.5, 0};

    A.trainOnce(inputs, desired);

    EXPECT_TRUE(A != B) << "A != B";

    B.trainOnce(inputs, desired);

    EXPECT_TRUE(A == B) << "A == B";

    A.trainOnce(inputs, desired);

    EXPECT_TRUE(A.getF1Score() == B.getF1Score()) << "A == B";
}

TEST(SaveNeuralNetwork, Save)
{
    StraightforwardNeuralNetwork A(5, {
        AllToAll(20, ReLU),
        AllToAll(10, snn::tanh),
        AllToAll(3, sigmoid)
    });
    A.learningRate = 0.03f;
    A.momentum = 0.78f;

    A.saveAs("./testSave.snn");

    StraightforwardNeuralNetwork B = StraightforwardNeuralNetwork::loadFrom("./testSave.snn");

    EXPECT_TRUE(A == B);
}
