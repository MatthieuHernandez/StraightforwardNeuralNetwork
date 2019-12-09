#include <stdexcept>

namespace snn::internal
{
	class NotImplementedException : public std::runtime_error
	{
	public:
		NotImplementedException() : std::runtime_error("Function not yet implemented")
		{
		}
	};
}