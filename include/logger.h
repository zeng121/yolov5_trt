#pragma once
#include<NvInfer.h>
#include <glog/logging.h>

class TRT_Logger : public nvinfer1::ILogger
{
    nvinfer1::ILogger::Severity _verbosity;
    std::ostream* _ostream;

public:
    TRT_Logger(Severity verbosity = Severity::kWARNING, std::ostream& ostream = std::cout)
        : _verbosity(verbosity)
        , _ostream(&ostream)
    {
    }
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= _verbosity)
        {
            time_t rawtime = std::time(0);
            char buf[256];
            strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", std::gmtime(&rawtime));
            const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR
                        ? "  ERROR"
                        : severity == Severity::kWARNING ? "WARNING" : severity == Severity::kINFO ? "   INFO"
                                                                                                   : "UNKNOWN");
            (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
        }
    }
};