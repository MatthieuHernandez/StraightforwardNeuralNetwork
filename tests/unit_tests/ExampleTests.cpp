#include "../ExtendedGTest.hpp"
#include "../../examples/ClassificationExample.cpp"
#include "../../examples/MultipleClassificationExemple.cpp"
#include "../../examples/RegressionExample.cpp"

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