#include <gtest/gtest.h>

#include "../ExtendedGTest.hpp"

auto main(int ac, char* av[]) -> int
{
    testing::InitGoogleTest(&ac, av);
    const auto tests = RUN_ALL_TESTS();
    ::testing::GTEST_FLAG(filter) = "Dataset_";
    return tests;
}
