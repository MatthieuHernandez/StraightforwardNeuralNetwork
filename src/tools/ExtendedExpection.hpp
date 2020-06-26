#pragma once
#include <stdexcept>

namespace snn
{
    class NotImplementedException : public std::runtime_error
    {
    public:
        NotImplementedException(std::string functionName = "Function")
            : std::runtime_error(functionName + " not yet implemented.")
        {
        }
    };

    class ShouldNeverBeCalledException : public std::runtime_error
    {
    public:
        ShouldNeverBeCalledException(std::string functionName = "Function")
            : std::runtime_error(functionName + " should never be called.")
        {
        }
    };

    class FileOpeningFailedException : public std::runtime_error
    {
    public:
        FileOpeningFailedException(std::string fileName = "file")
            : std::runtime_error("Cannot open " + fileName + ".")
        {
        }
    };

    class InvalidArchitectureException : public std::runtime_error
    {
    public:
        InvalidArchitectureException(std::string message)
            : std::runtime_error("Invalid neural network architecture: " + message)
        {
        }
    };
}
