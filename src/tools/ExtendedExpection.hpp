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

	class FileOpeningFailed : public std::runtime_error
	{
	public:
		FileOpeningFailed() : std::runtime_error("Cannot open file")
		{
		}
	};
}