#pragma once
#include <stdexcept>

namespace snn
{
class NotImplementedException : public std::runtime_error
{
    public:
        explicit NotImplementedException(const std::string& functionName = "Function")
            : std::runtime_error(functionName + " not yet implemented.")
        {
        }
};

class ShouldNeverBeCalledException : public std::runtime_error
{
    public:
        explicit ShouldNeverBeCalledException(const std::string& functionName = "Function")
            : std::runtime_error(functionName + " should never be called.")
        {
        }
};

class FileOpeningFailedException : public std::runtime_error
{
    public:
        explicit FileOpeningFailedException(const std::string& fileName = "file")
            : std::runtime_error("Cannot open " + fileName + ".")
        {
        }
};

class InvalidArchitectureException : public std::runtime_error
{
    public:
        explicit InvalidArchitectureException(const std::string& message)
            : std::runtime_error("Invalid neural network architecture: " + message)
        {
        }
};
}  // namespace snn
