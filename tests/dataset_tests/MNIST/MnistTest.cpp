#include "../../GTestTools.hpp"
#include "Mnist.hpp"

using namespace std;

class MnistTest : public testing::Test
{
protected :
    MnistTest()
    {
        dataset = new Mnist();
    }

public :
    Dataset* dataset;
};

TEST_F(MnistTest, loadData)
{
    ASSERT_FALSE(dataset->data);
}