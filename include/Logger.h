#ifndef LOGGER_H
#define LOGGER_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <string>
#include <mutex>

namespace AI {
namespace logs {

enum class LogLevel {
    Debug = 1,
    Info,
    Error
};

template<typename... Types>
class Logger {
public:
    Logger() = default;

    void setLogLevel(LogLevel level) {
        log_level_ = level;
    }

    void setAlarmLevel(LogLevel level) {
        alarm_level_ = level;
    }

    void setOutputFile(const std::string& filename) {
        output_file_.open(filename, std::ios::app);
    }

    template <typename... Args>
    void debug(const char* file, int line, const char* func, const Args&... args) {
        write(LogLevel::Debug, file, line, func, args...);
    }

    template <typename... Args>
    void info(const char* file, int line, const char* func, const Args&... args) {
        write(LogLevel::Info, file, line, func, args...);
    }

    template <typename... Args>
    void error(const char* file, int line, const char* func, const Args&... args) {
        write(LogLevel::Error, file, line, func, args...);
    }
    ~Logger() {}

private:
    LogLevel log_level_ = LogLevel::Debug; // 默认日志等级
    LogLevel alarm_level_ = LogLevel::Error; // 默认告警等级
    const std::string filename = "Logger.log"; // 默认日志输出文件
    std::ofstream output_file_{filename, std::ios::app};
    std::mutex mutex_;

    template <typename T>
    void writeArg(std::stringstream& ss, const T& arg) {
        ss << arg;
    }

    template <typename T, typename... Args>
    void writeArg(std::stringstream& ss, const T& arg, const Args&... args) {
        writeArg(ss, arg);
        ss << ", ";
        writeArg(ss, args...);
    }

    template <typename... Args>
    void write(LogLevel level, const char* file, int line, const char* func, const Args&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level >= log_level_) {
            std::time_t now = std::time(nullptr);
            std::tm *ltm = std::localtime(&now);
            char time_buffer[80];
            std::strftime(time_buffer, 80, "%Y-%m-%d %H:%M:%S", ltm);

            std::stringstream ss;
            writeArg(ss, args...);
            std::string message = ss.str();

            printf("[%s] [user] [%s] [%s:%d:%s] %s\n", time_buffer, levelStr(level).c_str(),
                   file, line, func, message.c_str());
            if (level >= alarm_level_) {
                printf(">>>> THIS IS AN ERROR! <<<<\n"); // 打印告警提示
            }
            if (output_file_.is_open()) {
                output_file_ << "[" << time_buffer << "] [AI] [" << levelStr(level) << "] "
                             << file << "(" << line << "): " << message << std::endl;
            }
        }
    }

    std::string levelStr(LogLevel level) {
        switch (level) {
        case LogLevel::Debug:
            return "debug";
        case LogLevel::Info:
            return "info";
        case LogLevel::Error:
            return "error";
        }
        return "";
    }
};

static Logger<> logger;

#define LOG_DEBUG(...) AI::logs::logger.debug(__FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_INFO(...) AI::logs::logger.info(__FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_ERROR(...) AI::logs::logger.error(__FILE__, __LINE__, __func__, __VA_ARGS__)

#endif // LOGGER_H
}
}