#include <iostream>
#include <string>
#include "ExtendedGTest.hpp"

using namespace std;

template<typename T>
void EXPECT_ABOUT_EQ(T min, T value, T max, string valueName)
{
    ASSERT_TRUE(value >= min) << valueName + " = " + to_string(value) + "\t\t >= " + to_string(min);
    ASSERT_TRUE(value <= max) << valueName + " = " + to_string(value) + "\t\t <= " + to_string(max);
}

void PRINT_LOG(string str)
{
    cout << "\033[0;32m" << "[          ] " << "\033[0;0m" << str << endl;
}

