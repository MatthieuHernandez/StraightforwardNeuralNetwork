#include "../ExtendedGTest.hpp"

TEST(TestCaseName, Test_1) { EXPECT_EQ(1, 1); }

TEST(TestCaseName, Test_2) { EXPECT_TRUE(true) << "message test"; }