#include "../src/straightforwardNeuralNetwork.h"
using snn;
/*
* This is the simpliest example how to use this library*
* In this neural network return 3 ouputs AND, NAND, OR, XOR logical operator of 2 inputs.
* For more explaination go to wiki
*/
int main()
{
	std::vector<std::vector<float>> inputData = {{0,0},{0,1},{1,0},{1,1}};
	std::vector expectedOutput = {{0, 1, 0, 0},
								  {0, 1, 1, 1}, 
								  {0, 1, 1, 1},
								  {1, 0, 1, 0}};
								
	DataInput data(inputData, expectedOutput);
	
	StraightforwardNeuralNetwork neuralNetwork(std::vector<float> {2, 10, 10, 4},
											   std::vector<float> {Sigmoid, Sigmoid, Sigmoid});
											   
	neuralNetwork.train(data);
	Sleep(5000); // train neural network during 5 seconds
	float accuracy = neuralNetwork.getClusteringRate();
	printf("accuracy = %f.2", accuracy);
	
	vector<float> output = neuralNetwork.computeOutput(inputData[0]) // consult neural network to test it
	
	if(output == expectedOutput[0]) 
		return EXIT_SUCCESS; 
	else
		return EXIT_FAILURE;
}