#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

namespace surprise {

class SurpriseManager {
public:
  SurpriseManager(int threshold = 128,
                  double alpha     = 0.6,
                  double beta      = 0.3,
                  double gamma     = 0.1,
                  int beam_width   = 5);

  std::vector<int> surprise(
      const std::vector<std::vector<uchar>>& slices
  ) const;

  // SSIM constants
  static constexpr double K1 = 0.01;
  static constexpr double K2 = 0.05;
  static constexpr double L  = 255.0;

private:
  // for larger N, use beam search
  std::vector<int> beamSearch(const std::vector<cv::Mat>& raws) const;

  cv::Mat   thresholdImage(const cv::Mat& img) const;
  double    ssim1d        (const cv::Mat& x, const cv::Mat& y) const;
  double    crossCorr     (const cv::Mat& x, const cv::Mat& y) const;

  static std::pair<double, std::vector<int>>
    findMaxWeightHamiltonianPath(const cv::Mat& weights,
                                 int start_node = -1);
  static std::pair<double, std::vector<int>>
    findMaxWeightHamiltonianCycle(const cv::Mat& weights,
                                  int start_node);

  int   threshold_;
  double alp, bet, gam;
  int   beamWidth_;
};

}
