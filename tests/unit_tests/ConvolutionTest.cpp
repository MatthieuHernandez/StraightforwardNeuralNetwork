#include "../ExtendedGTest.hpp"
#include "tools/ExtendedExpection.hpp"
#include "tools/Tools.hpp"
#include "neural_network/StraightforwardNeuralNetwork.hpp"

using namespace std;
using namespace snn;

DataForClassification CreateData();

TEST(Architecture, ValidArchitectures)
{
    auto data = CreateData();
    vector2D<LayerModel> architectures =
    {
        {Input(3, 3, 2), Convolution(2, 2, sigmoid)},
        {AllToAll(1)}
    };
    StraightforwardNeuralNetwork nn(architectures);
    nn.startTraining(data);
    nn.waitFor(20_ep);
    nn.stopTraining();
    float accuracy = neuralNetwork.getGlobalClusteringRate();
    ASSERT_ACCURACY(accuracy, 1.0)

}

DataForClassification CreateData()
{
    vector<vector<float>> inputData = {
        {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 
         0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
         0.00, 0.00, 0.00, 0.00, 0.00, 0.00},

        {0.11, 0.12, 0.21, 0.22, 0.31, 0.32, 
         0.41, 0.42, 0.51, 0.52,  0.61, 0.62
         0.71, 0.72, 0.81, 0.82, 0.91, 0.92}
    };
	vector2D<float>> expectedOutputs = {{0}, {1}};

	DataForClassification data(inputData, expectedOutputs);
}