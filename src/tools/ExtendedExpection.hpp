#include <stdexcept>

namespace snn::internal
{
	class NotImplementedException : public std::runtime_error
	{
	public:
		NotImplementedException(string functionName = "Function") : std::runtime_error(functionName + " not yet implemented")
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