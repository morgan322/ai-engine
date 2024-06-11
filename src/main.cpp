#include "Logger.h"
#include "class_detector.h"
#include "class_timer.hpp"

#include <memory>
#include <thread>
// #include "mainwindow.h"
// #include <QApplication>

int main(int argc, char *argv[]) {
  Config config_v5;
  config_v5.net_type = YOLOV5;
  config_v5.detect_thresh = 0.5;
  config_v5.file_model_cfg = "/media/ai/AI/technology/computer/ai-engine/"
                             "config/yolov5-6.0/yolov5s.cfg";
  config_v5.file_model_weights = "/media/ai/AI/technology/computer/ai-engine/"
                                 "config/yolov5-6.0/yolov5s.weights";
  config_v5.calibration_image_list_file_txt =
      "/media/ai/AI/technology/computer/ai-engine/config/"
      "calibration_images.txt";
  config_v5.inference_precison = FP16; // FP32 FP16 INT8

  std::unique_ptr<Detector> detector(new Detector());
  detector->init(config_v5);
  cv::Mat image0 =
      cv::imread("/media/ai/AI/technology/computer/ai-engine/data/dog.jpg",
                 cv::IMREAD_UNCHANGED);
  cv::Mat image1 =
      cv::imread("/media/ai/AI/technology/computer/ai-engine/data/person.jpg",
                 cv::IMREAD_UNCHANGED);
  std::vector<BatchResult> batch_res;
  // Timer timer;
  // for (;;)
  // while(true)
  // {
  // prepare batch data
  std::vector<cv::Mat> batch_img;
  cv::Mat temp0 = image0.clone();
  // cv::Mat temp1 = image1.clone();
  batch_img.push_back(temp0);
  // batch_img.push_back(temp1);

  // detect
  //  timer.reset();
  detector->detect(batch_img, batch_res);
  // timer.out("detect");

  // disp
  while (1) {
    for (int i = 0; i < batch_img.size(); ++i) {

      for (const auto &r : batch_res[i]) {
        // std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << "
        // rect:" << r.rect << std::endl;
        LOG_INFO(r.id, r.prob, r.rect);
        cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << "id:" << r.id
               << "  score:" << r.prob;
        cv::putText(batch_img[i], stream.str(),
                    cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5,
                    cv::Scalar(0, 0, 255), 2);
      }
      cv::namedWindow("image" + std::to_string(i), cv::WINDOW_NORMAL);
      cv::imshow("image" + std::to_string(i), batch_img[i]);
      cv::waitKey(10);
    }
  }

  //
  // }
  // QApplication a(argc, argv);
  // MainWindow w;
  // w.setWindowTitle("Canny Edge Detection");
  // w.show();
  // return a.exec();

  return 0;
}