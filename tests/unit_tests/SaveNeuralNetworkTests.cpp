#include "../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

TEST(SaveNeuralNetwork, EqualTest)
{
    auto structureOfNetwork = {
        Input(8, 8, 3),
        Convolution(2, 4, ReLU),
        AllToAll(20, iSigmoid),
        Convolution(3, 2, snn::tanh),
        AllToAll(3, sigmoid)
    };
    StraightforwardNeuralNetwork A(structureOfNetwork);
    StraightforwardNeuralNetwork C(structureOfNetwork);
    A.optimizer.learningRate = 0.03f;
    A.optimizer.momentum = 0.78f;
    C.optimizer.learningRate = 0.03f;
    C.optimizer.momentum = 0.78f;
    StraightforwardNeuralNetwork B = A;

    ASSERT_EQ(A.isValid(), 0) << "A is invalid";
    ASSERT_EQ(B.isValid(), 0) << "B is invalid";

    EXPECT_TRUE(A == B);
    EXPECT_TRUE(&A != &B);

    EXPECT_TRUE(&A != &B) << "&A != &B";


    EXPECT_TRUE(A.optimizer == B.optimizer) << "Value : A.optimiser == B.optimiser";
    EXPECT_TRUE(&A.optimizer != &B.optimizer) << "Address : A.optimiser != B.optimiser";

    EXPECT_TRUE(*A.layers[0] == *B.layers[0]) << "Value : A.layers[0] == B.layers[0]";
    EXPECT_TRUE(A.layers[0] != B.layers[0]) << "Address : A.layers[0] != B.layers[0]";


    EXPECT_TRUE(*&A.layers[0]->neurons[0] == *&B.layers[0]->neurons[0]) <<
 "Value : A.Layers[0].neurons[0] == B.Layers[0].neurons[0]";
    EXPECT_TRUE(&A.layers[0]->neurons[0] != &B.layers[0]->neurons[0]) <<
 "Address : A.Layers[0].neurons[0] != B.Layers[0].neurons[0]";

    EXPECT_TRUE(&A.optimizer == &*A.layers[0]->neurons[0].optimizer);
    EXPECT_TRUE(&B.optimizer == &*B.layers[0]->neurons[0].optimizer);

    EXPECT_TRUE(A != C); // Test A == C with same seed

    vector<float> inputs(8*8*3);
    inputs[30] = 1.5;
    inputs[60] = 0.75;
    inputs[90] = -0.25;
    inputs[120] = 1;
    inputs[150] = -1.35;
    const vector<float> desired{1, 0, 0.5, 0};

    A.trainOnce(inputs, desired);

    EXPECT_TRUE(A != B) << "A != B";

    B.trainOnce(inputs, desired);

    EXPECT_TRUE(A == B) << "A == B";

    A.trainOnce(inputs, desired);

    EXPECT_TRUE(A.getF1Score() == B.getF1Score()) << "A == B";
}

TEST(SaveNeuralNetwork, Save)
{
    StraightforwardNeuralNetwork A({
        Input(5),
        Convolution(2, 2, ReLU),
        AllToAll(10, snn::tanh),
        AllToAll(3, sigmoid)
    });
    A.optimizer.learningRate = 0.03f;
    A.optimizer.momentum = 0.78f;

    A.saveAs("./testSave.snn");

    StraightforwardNeuralNetwork B = StraightforwardNeuralNetwork::loadFrom("./testSave.snn");
    EXPECT_TRUE(A == B);
    EXPECT_TRUE(B.isValid() == 0);
}
