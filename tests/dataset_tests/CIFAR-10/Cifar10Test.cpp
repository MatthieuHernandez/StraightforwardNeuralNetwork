#include "../../GTestTools.hpp"
#include "Cifar10.hpp"

using namespace std;

class Cifar10Test : public testing::Test
{
protected:
    Cifar10Test()
    {
        dataset = new Cifar10();
    }

public:
    Dataset* dataset;
};

TEST_F(Cifar10Test, loadData)
{
    ASSERT_FALSE(dataset->data);
}