#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_bool(verbose, false, "Verbose mode");
DEFINE_int32(m, 0, "Message limit");
DEFINE_int32(c, 1, "Custom parameter");

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::SetStderrLogging(google::GLOG_INFO);

    if (FLAGS_verbose) {
        LOG(INFO) << "Verbose mode is enabled.";
    } else {
        LOG(INFO) << "Verbose mode is disabled.";
    }

    LOG(INFO) << "Message limit: " << FLAGS_m;
    LOG(INFO) << "Custom parameter: " << FLAGS_c;

    google::ShutdownGoogleLogging();
    return 0;
}
