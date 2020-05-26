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

TEST(exemples, simpleRecurrence)
{
    ASSERT_EQ(simpleRecurrenceExample(), EXIT_SUCCESS);
}

TEST(exemples, mediumRecurrence)
{
    ASSERT_EQ(mediumRecurrenceExample(), EXIT_SUCCESS);
}

TEST(exemples, complexRecurrence)
{
    ASSERT_EQ(complexRecurrenceExample(), EXIT_SUCCESS);
}