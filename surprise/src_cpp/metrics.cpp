#include "metrics.h"
#include <numeric>

namespace metrics {

static void computeMeanVar(
    const cv::Mat& M,
    cv::Mat& mean,
    cv::Mat& var
) {
    int N = M.rows, D = M.cols;
    mean.create(N,1,CV_64F);
    var .create(N,1,CV_64F);
    for (int i = 0; i < N; ++i) {
        const double* row = M.ptr<double>(i);
        double μ = std::accumulate(row, row+D, 0.0) / D;
        mean.at<double>(i,0) = μ;
        double s = 0;
        for (int j = 0; j < D; ++j) {
            double d = row[j] - μ;
            s += d*d;
        }
        var.at<double>(i,0) = s / D;
    }
}

cv::Mat calculateSSIMBatchVsBatch(
    const cv::Mat& edges_batch1,
    const cv::Mat& edges_batch2,
    double L
) {
    cv::Mat A = edges_batch1, B = edges_batch2;
    if (A.type() != CV_64F) A.convertTo(A, CV_64F);
    if (B.type() != CV_64F) B.convertTo(B, CV_64F);

    int N1 = A.rows, N2 = B.rows, D = A.cols;

    const double K1 = 0.01, K2 = 0.03;
    const double C1 = (K1*L)*(K1*L);
    const double C2 = (K2*L)*(K2*L);

    cv::Mat mu1, var1, mu2, var2;
    computeMeanVar(A, mu1, var1);
    computeMeanVar(B, mu2, var2);

    cv::Mat A_c = A.clone();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < D; ++j)
            A_c.at<double>(i,j) -= mu1.at<double>(i,0);

    cv::Mat B_c = B.clone();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N2; ++i)
        for (int j = 0; j < D; ++j)
            B_c.at<double>(i,j) -= mu2.at<double>(i,0);

    // compute cross-covariance: (A_c row i)·(B_c row j)^T / D
    // result sigma12[i,j]
    cv::Mat sigma12(N1, N2, CV_64F);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            double sum = 0;
            for (int k = 0; k < D; ++k)
                sum += A_c.at<double>(i,k) * B_c.at<double>(j,k);
            sigma12.at<double>(i,j) = sum / D;
        }
    }

    // compute SSIM matrix
    cv::Mat ssim(N1, N2, CV_64F);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            double μ1 = mu1.at<double>(i,0),
                   μ2 = mu2.at<double>(j,0),
                   v1 = var1.at<double>(i,0),
                   v2 = var2.at<double>(j,0),
                   cov = sigma12.at<double>(i,j);

            double num = (2*μ1*μ2 + C1) * (2*cov + C2);
            double den = (μ1*μ1 + μ2*μ2 + C1) * (v1 + v2 + C2);
            ssim.at<double>(i,j) = den == 0 ? 1.0 : num/den;
        }
    }

    return ssim;
}

}
