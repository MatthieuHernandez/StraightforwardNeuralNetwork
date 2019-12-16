#include "../GTestTools.hpp"
#include "../../examples/ClassificationExample.cpp"
#include "../../examples/MultipleClassificationExemple.cpp"
#include "../../examples/RegressionExample.cpp"

TEST(Exemples, classification)
{
    ASSERT_EQ(classificationExample(), EXIT_SUCCESS);
}

TEST(Exemples, multipleClassification)
{
    ASSERT_EQ(multipleClassificationExample(), EXIT_SUCCESS);
}

TEST(Exemples, regression)
{
    ASSERT_EQ(regressionExample(), EXIT_SUCCESS);
}