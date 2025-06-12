#pragma once
#include <opencv2/opencv.hpp>

namespace metrics {

cv::Mat calculateSSIMBatchVsBatch(
    const cv::Mat& edges_batch1, 
    const cv::Mat& edges_batch2,
    double L = 255.0
);

}
