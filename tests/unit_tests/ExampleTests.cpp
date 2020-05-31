#include "../ExtendedGTest.hpp"
#include "../../examples/Examples.hpp"

TEST(exemples, classification)
{
    ASSERT_EQ(classificationExample(), EXIT_SUCCESS);
}

TEST(exemples, multipleClassification)
{
    ASSERT_EQ(multipleClassificationExample(), EXIT_SUCCESS);
}

TEST(exemples, regression)
{
    ASSERT_EQ(regressionExample(), EXIT_SUCCESS);
}

TEST(exemples, recurrenceExample)
{
    ASSERT_EQ(recurrenceExample(), EXIT_SUCCESS);
}