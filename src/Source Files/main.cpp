#include "neuralNetwork.h"
#include <iostream>

//#define PRECISION 1e-6f // 2e-7f

int main()
{
    std::cout << "IT COMPILES !" << std::endl << std::endl;
    return 0;
}
    /*NeuralNetwork testCPU(4, 4, 100, 2);
    NeuralNetwork testGPU(1, 1, 1, 1);
    testCPU.setLearningRate(0.7f);

    testGPU = testCPU;
    testGPU.setUseGPU(true);

    if(testCPU.isValid() != 0
    || testGPU.isValid() != 0)
    {
        cout << "ERROR : " << testCPU.isValid() << endl;
    }
    vector<float> input;
    input.push_back(4.316f);
    input.push_back(-7.264f);
    input.push_back(0.257f);
    input.push_back(-0.944f);

    vector<float> output_GPU;
    output_GPU.resize(2);

    vector<float> output_CPU;
    output_CPU.resize(2);

    output_CPU = testCPU.outputFloat(input);
    output_GPU = testGPU.outputFloat(input);

    if(output_CPU[0] < output_GPU[0] + PRECISION && output_CPU[0] > output_GPU[0] - PRECISION
    && output_CPU[1] < output_GPU[1] + PRECISION && output_CPU[1] > output_GPU[1] - PRECISION
    && (output_GPU[0] != 0 || output_GPU[1] != 0))
    {
        cout << "Output calculation WORKS !!!" << endl;
    }
    else
    {
        cout << "Sorry, output calculation doesn't work." << endl;
        cout << setprecision(50) << output_CPU[0] << " != " << output_GPU[0] << endl;
        cout << setprecision(50) << output_CPU[1] << " != " << output_GPU[1] << endl;
    }
    cout << endl;

    vector<float> desired;
    desired.push_back(0.5);
    desired.push_back(0.5);
    desired.push_back(0.5);
    desired.push_back(0.5);


    for(int i = 0; i < 1; i++)
    {
        testCPU.train(input, desired);
        testGPU.train(input, desired);
    }

    output_CPU = testCPU.outputFloat(input);
    output_GPU = testGPU.outputFloat(input);

    if(output_CPU[0] < output_GPU[0] + PRECISION && output_CPU[0] > output_GPU[0] - PRECISION
    && output_CPU[1] < output_GPU[1] + PRECISION && output_CPU[1] > output_GPU[1] - PRECISION
    && (output_GPU[0] != 0 || output_GPU[1] != 0))
    {
        cout << "Backpropation calculation WORKS !!!" << endl;
    }
    else
    {
        cout << "Sorry, backropagation doesn't work." << endl;
        cout << setprecision(50) << output_CPU[0] << " != " << output_GPU[0] << endl;
        cout << setprecision(50) << output_CPU[1] << " != " << output_GPU[1] << endl;
    }*/
