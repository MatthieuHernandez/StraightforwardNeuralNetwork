#include "../ExtendedGTest.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"
#include "neural_network/layer/neuron/Neuron.hpp"

using namespace std;
using namespace snn;

TEST(SaveNeuralNetwork, EqualTest)
{
    auto structureOfNetwork = {
        Input(8, 8, 3),
        Convolution(2, 4, activation::ReLU),
        FullyConnected(20, activation::iSigmoid),
        LocallyConnected(3, 2, activation::tanh),
        FullyConnected(3, activation::sigmoid),
        GruLayer(3),
        Recurrence(3)
    };
    StraightforwardNeuralNetwork A(structureOfNetwork);
    StraightforwardNeuralNetwork C(structureOfNetwork);
    A.optimizer.learningRate = 0.03f;
    A.optimizer.momentum = 0.78f;
    C.optimizer.learningRate = 0.03f;
    C.optimizer.momentum = 0.78f;
    StraightforwardNeuralNetwork B = A;

    ASSERT_EQ(A.isValid(), 0);
    ASSERT_EQ(B.isValid(), 0);

    EXPECT_TRUE(A == B);
    EXPECT_TRUE(&A != &B);

    EXPECT_TRUE(&A != &B);


    EXPECT_TRUE(A.optimizer == B.optimizer);
    EXPECT_TRUE(&A.optimizer != &B.optimizer);

    EXPECT_TRUE(*A.layers[0] == *B.layers[0]);
    EXPECT_TRUE(A.layers[0] != B.layers[0]);

    auto* neuronA = static_cast<internal::SimpleNeuron*>(A.layers[0]->getNeuron(0));
    auto* neuronB = static_cast<internal::SimpleNeuron*>(B.layers[0]->getNeuron(0));
    EXPECT_TRUE(*neuronA == *neuronB);
    EXPECT_TRUE(neuronA != neuronB);

    //auto moto = dynamic_cast<internal::SimpleNeuron>(A.layers[0]->getNeuron(0));
    EXPECT_TRUE(&A.optimizer == &*static_cast<internal::SimpleNeuron*>(A.layers[0]->getNeuron(0))->optimizer);
    EXPECT_TRUE(&B.optimizer == &*static_cast<internal::SimpleNeuron*>(B.layers[0]->getNeuron(0))->optimizer);

    EXPECT_TRUE(A != C); // Test A == C with same seed

    vector<float> inputs(8*8*3);
    inputs[30] = 1.5f;
    inputs[60] = 0.75f;
    inputs[90] = -0.25f;
    inputs[120] = 1.0f;
    inputs[150] = -1.35f;
    const vector<float> desired{1.0f, 0.0f, 0.5f, 0.07f};

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
        Input(15),
        Convolution(2, 2, activation::ReLU),
        LocallyConnected(2, 2, activation::tanh),
        FullyConnected(3, activation::sigmoid),
        GruLayer(2)
    });
    A.optimizer.learningRate = 0.03f;
    A.optimizer.momentum = 0.78f;

    A.saveAs("./testSave.snn");

    StraightforwardNeuralNetwork B = StraightforwardNeuralNetwork::loadFrom("./testSave.snn");
    EXPECT_TRUE(A == B);
    EXPECT_TRUE(B.isValid() == 0);
}
