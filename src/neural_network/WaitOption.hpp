#include <iostream>
using namespace std;

enum waitType {epoches = 0, accuracy = 1};

template<typename T>
struct WaitOption
{
    waitType type;
    T value;
    operator||(const WaitOption<T>& option)
    {
        
    }
};

WaitOption<int> operator""_ep (unsigned long long value) { 
    return WaitOption<int>{epoches, (int)value}; 
    
}
WaitOption<float> operator""_acc (long double value) { 
    return WaitOption<float>{accuracy, (float)value}; 
    
}
class NeuralNetwork
{
public:

    void waitFor(WaitOption<int> option)
    {
        cout << "Wait for " << option.value << " epoches" << endl;
    };
    
    void waitFor(WaitOption<float> option)
    {
        cout << "Wait for " << option.value << " accuracy" << endl;
    };

};

int main() { 
    NeuralNetwork nn;
    nn.waitFor(10_ep);
    nn.waitFor(0.9_acc);
}