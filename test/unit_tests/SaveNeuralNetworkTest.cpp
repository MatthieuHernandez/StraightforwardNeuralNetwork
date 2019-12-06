#include "GTestTools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;
// ReSharper disable CppInconsistentNaming CppLocalVariableMayBeConst CppUseAuto

TEST(SaveNeuralNetwork, EqualTest)
{
	const vector<int> structureOfNetwork {5, 20, 10, 3};
	const vector<activationFunctionType> activationFunctionByLayer{iSigmoid, tanH, sigmoid};
	StraightforwardOption optionA;
	optionA.learningRate = 0.03f;
	optionA.momentum = 0.78f;
	StraightforwardOption optionC;
	optionC.learningRate = 0.03f;
	optionC.momentum = 0.78f;
	StraightforwardNeuralNetwork A(structureOfNetwork, activationFunctionByLayer, optionA);
	StraightforwardNeuralNetwork C(structureOfNetwork, activationFunctionByLayer, optionC);
	StraightforwardNeuralNetwork B = A;

	ASSERT_EQ(A.isValid(), 0) << "A is invalid";
	ASSERT_EQ(B.isValid(), 0) << "B is invalid";

	EXPECT_TRUE(A == B);
	EXPECT_TRUE(&A != &B);

	EXPECT_TRUE(&A != &B) << "&A != &B";

	EXPECT_TRUE(A.option == B.option) << "A.option == B.option";
	EXPECT_TRUE(&(A.option) != &(B.option)) << "A.option != B.option";

	//EXPECT_TRUE(*A.getLayer(0) == *B.getLayer(0)) << "Value : A.layers[0] == B.layers[0]";
	//EXPECT_TRUE(A.getLayer(0) != B.getLayer(0)) << "Address : A.layers[0] != B.layers[0]";

	//EXPECT_TRUE(*&A.getLayer(0).getNeuron(0) == *&B.getLayer(0).getNeuron(0)) << "Value : A.Layers[0].neurons[0] == B.Layers[0].neurons[0]";
	//EXPECT_TRUE(&A.getLayer(0).getNeuron(0) != &B.getLayer(0).getNeuron(0)) << "Address : A.Layers[0].neurons[0] != B.Layers[0].neurons[0]";
	
	//EXPECT_TRUE(*A.getLayer(0).getNeuron(0).getActivationFunction() == *B.getLayer(0).getNeuron(0).getActivationFunction()) << "Value : A.Layers[0].neurons[0] == B.Layers[0].neurons[0]";
	//EXPECT_TRUE(A.getLayer(0).getNeuron(0).getActivationFunction() != B.getLayer(0).getNeuron(0).getActivationFunction()) << "Address : A.Layers[0].neurons[0] != B.Layers[0].neurons[0]";

	EXPECT_TRUE(A != C); // Test A == C with same seed

	/*const vector<float> inputs {1.5, 0.75, -0.25, 0, 0};
	const vector<float> desired {1, 0, 0.5, 0};

	A.trainOnce(inputs, desired);

	EXPECT_TRUE(A != B) << "A != B";

	B.trainOnce(inputs, desired);

	EXPECT_TRUE(A == B) << "A == B";

	A.trainOnce(inputs, desired);

	EXPECT_TRUE(A.getF1Score() == B.getF1Score()) << "A == B";*/
}

TEST(SaveNeuralNetwork, Save)
{
	const vector<int> structureOfNetwork {5, 20, 10, 3};
	const vector<activationFunctionType> activationFunctionByLayer{iSigmoid, tanH, sigmoid};
	StraightforwardOption option;
	option.learningRate = 0.03f;
	option.momentum = 0.78f;
	StraightforwardNeuralNetwork A(structureOfNetwork, activationFunctionByLayer, option);

	A.saveAs("./testSave.snn");

	StraightforwardNeuralNetwork B = StraightforwardNeuralNetwork::loadFrom("./testSave.snn");

	//EXPECT_TRUE(A == B) << "A == B";
}
