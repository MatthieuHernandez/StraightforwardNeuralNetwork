#include "../../GTestTools.hpp"
#include "Iris.hpp"

using namespace std;

class IrisTest : public testing::Test
{
protected:
    IrisTest()
    {
        dataset = new Iris();
    }

public:
    Dataset* dataset;
};

TEST_F(IrisTest, loadData)
{
    ASSERT_TRUE(dataset->data);
}