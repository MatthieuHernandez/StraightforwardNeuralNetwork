 #pragma once
#include "activation_function/ActivationFunction.hpp"

 namespace snn
{
     struct NeuronModel
     {
         int numberOfInputs;
         int numberOfWeights;
         int numberOfRecurrences;
         activationFunction activation;
     };
 }
