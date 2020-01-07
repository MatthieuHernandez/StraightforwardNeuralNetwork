#include "../ExtendedGTest.hpp"
#include "../../examples/ClassificationExample.cpp"
#include "../../examples/MultipleClassificationExemple.cpp"
#include "../../examples/RegressionExample.cpp"

TEST(exemples, DISABLED_classification)
{
    ASSERT_EQ(classificationExample(), EXIT_SUCCESS);
}

TEST(exemples, DISABLED_multipleClassification)
{
    ASSERT_EQ(multipleClassificationExample(), EXIT_SUCCESS);
}

TEST(exemples, DISABLED_regression)
{
    ASSERT_EQ(regressionExample(), EXIT_SUCCESS);
}