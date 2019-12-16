#include "../../GTestTools.hpp"
#include "data/Data.hpp"
#include "Cifar10.hpp"

using namespace std;
using namespace snn;

class Cifar10Test : public testing::Test
{
protected :
    Cifar10Test()
    {
        dataset = new Cifar10();
    }

public :
    Dataset* dataset;
};

TEST_F(Cifar10Test, loadData)
{
    ASSERT_TRUE(dataset->data);
}