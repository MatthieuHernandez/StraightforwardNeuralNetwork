#include "ExtendedGTest.hpp"

#include <chrono>
#include <iostream>
#include <sstream>

using namespace std;
using namespace chrono;

template <typename T>
void ASSERT_BETWEEN(T min, T value, T max, string valueName)
{
    ASSERT_GE(value, min);
    ASSERT_LE(value, max);
}

void ASSERT_SUCCESS() { ASSERT_TRUE(true); }

void ASSERT_FAIL(string message) { ASSERT_TRUE(false) << message; }

void ASSERT_VECTOR_EQ(const vector<float> values, const vector<float> expected_values)
{
    ASSERT_EQ(values.size(), expected_values.size());
    for (auto i = 0; i < values.size(); ++i) ASSERT_NEAR(values[i], expected_values[i], 1e-6F);
}

void PRINT_LOG(string message) { cout << "[          ] " << message << endl; }

void PRINT_NUMBER_OF_NEURONS(int value) { cout << "[          ] " << value << " neurons" << endl; }

void PRINT_NUMBER_OF_PARAMETERS(int value) { cout << "[          ] " << value << " parameters" << endl; }

void PRINT_RESULT(string message) { cout << "[ RESULT   ] " << message << endl; }

void ASSERT_ACCURACY(float actual, float expected)
{
    stringstream message;
    message << "Accuracy = " << std::fixed << std::setprecision(2) << actual * 100.0F << "%";
    PRINT_RESULT(message.str());
    ASSERT_GE(actual, expected);
}

void ASSERT_RECALL(float actual, float expected)
{
    stringstream message;
    message << "Recall = " << std::fixed << std::setprecision(2) << actual * 100.0F << "%";
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
