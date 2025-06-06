#include "ExtendedGTest.hpp"

#include <iostream>
#include <sstream>

template <typename T>
static void ASSERT_BETWEEN(T min, T value, T max)
{
    ASSERT_GE(value, min);
    ASSERT_LE(value, max);
}

void ASSERT_SUCCESS() { ASSERT_TRUE(true); }

void ASSERT_FAIL(std::string message) { ASSERT_TRUE(false) << message; }

void ASSERT_VECTOR_EQ(const std::vector<float>& values, const std::vector<float>& expected_values, float abs_error)
{
    ASSERT_EQ(values.size(), expected_values.size());
    for (auto i = 0; i < values.size(); ++i)
    {
        ASSERT_NEAR(values[i], expected_values[i], abs_error) << "First mismatch at index " << i << ".";
    }
}

void PRINT_LOG(std::string message) { std::cout << "[          ] " << message << '\n'; }

void PRINT_NUMBER_OF_NEURONS(int value) { std::cout << "[          ] " << value << " neurons\n"; }

void PRINT_NUMBER_OF_PARAMETERS(int value) { std::cout << "[          ] " << value << " parameters\n"; }

void PRINT_RESULT(std::string message) { std::cout << "[ RESULT   ] " << message << '\n'; }

void ASSERT_ACCURACY(float actual, float expected)
{
    std::stringstream message;
    message << "Accuracy = " << std::fixed << std::setprecision(2) << actual * 100.0F << "%";
    PRINT_RESULT(message.str());
    ASSERT_GE(actual, expected);
}

void ASSERT_RECALL(float actual, float expected)
{
    std::stringstream message;
    message << "Recall = " << std::fixed << std::setprecision(2) << actual * 100.0F << "%";
    PRINT_RESULT(message.str());
    ASSERT_GE(actual, expected);
}

void ASSERT_MAE(float actual, float expected)
{
    std::stringstream message;
    message << "Mean Absolute Error = " << std::fixed << std::setprecision(2) << actual;
    PRINT_RESULT(message.str());
    ASSERT_LE(actual, expected);
}
