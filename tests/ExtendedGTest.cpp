#include <iostream>
#include <sstream>
#include "ExtendedGTest.hpp"

using namespace std;

template<typename T>
void ASSERT_BETWEEN(T min, T value, T max, string valueName)
{
    ASSERT_GE(value, min);
    ASSERT_LE(value, max);
}

void ASSERT_SUCCESS()
{
    ASSERT_TRUE(true);
}

void ASSERT_FAIL(string message)
{
    ASSERT_TRUE(false) << message;
}

void PRINT_LOG(string message)
{
    cout << "\033[0;32m" << "[          ]" << "\033[0;0m " << message << endl;
}

void PRINT_RESULT(string message)
{
    cout << "\033[0;32m" << "[ RESULT   ]" << "\033[0;0m " << message << endl;
}

void ASSERT_ACCURACY(float actual, float expected)
{
    stringstream message;
    message << "Accuracy = " << std::fixed << std::setprecision(2) << actual * 100.0f << "%";
    PRINT_RESULT(message.str());
    ASSERT_GE(actual, expected);
}

void ASSERT_MAE(float actual, float expected)
{
    stringstream message;
    message << "Mean Absolute Error = " << std::fixed << std::setprecision(2) << actual;
    PRINT_RESULT(message.str());
    ASSERT_LE(actual, expected);
}
