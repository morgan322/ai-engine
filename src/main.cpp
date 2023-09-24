#include "Logger.h"


int main() {
  
    LOG_DEBUG("This is a debug message"); // 输出 debug 日志
    LOG_INFO("This is an info message: ", "hello", 123, 3.14159); // 输出 info 日志
    LOG_ERROR("This is an error message: ", "something went wrong"); // 输出 error 日志
    
    return 0;
}