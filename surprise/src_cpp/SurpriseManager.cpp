#include "SurpriseManager.h"
#include "metrics.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>
// #include <iostream>

using namespace surprise;

SurpriseManager::SurpriseManager(int threshold, double alpha, double beta, double gamma, int beamWidth)
  : threshold_(threshold),
    alp(alpha), bet(beta), gam(gamma),
    beamWidth_(beamWidth)
{}

cv::Mat SurpriseManager::thresholdImage(const cv::Mat &img) const {
    cv::Mat bin;
    cv::threshold(img, bin, threshold_, 255, cv::THRESH_BINARY);
    return bin;
}

double SurpriseManager::ssim1d(const cv::Mat &x, const cv::Mat &y) const {
    // both x,y are single-row Mats of type CV_64F
    int D = x.cols;
    const double *xp = x.ptr<double>();
    const double *yp = y.ptr<double>();

    // means
    double mux = std::accumulate(xp, xp+D, 0.0)/D;
    double muy = std::accumulate(yp, yp+D, 0.0)/D;

    // variances & covariance
    double sigx2 = 0, sigy2 = 0, cov=0;
    for(int i=0;i<D;++i){
        double dx = xp[i]-mux, dy = yp[i]-muy;
        sigx2 += dx*dx;
        sigy2 += dy*dy;
        cov += dx*dy;
    }
    sigx2 /= D; sigy2 /= D; cov /= D;
    double C1 = (K1*L)*(K1*L), C2 = (K2*L)*(K2*L);
    double num = (2*mux*muy + C1)*(2*cov + C2);
    double den = (mux*mux + muy*muy + C1)*(sigx2 + sigy2 + C2);
    return den==0 ? 1.0 : num/den;
}

double SurpriseManager::crossCorr(const cv::Mat &x, const cv::Mat &y) const {
    int D = x.cols;
    const double *xp = x.ptr<double>(), *yp = y.ptr<double>();
    double mux = std::accumulate(xp, xp+D, 0.0)/D;
    double muy = std::accumulate(yp, yp+D, 0.0)/D;
    double num=0, sx=0, sy=0;
    for(int i=0;i<D;++i){
        double dx = xp[i]-mux, dy=yp[i]-muy;
        num += dx*dy;
        sx  += dx*dx;
        sy  += dy*dy;
    }
    double denom = std::sqrt(sx*sy);
    return denom==0 ? 0.0 : num/denom;
}

std::vector<int> SurpriseManager::beamSearch(const std::vector<cv::Mat> &raws) const {
    int N = raws.size();

    std::vector<cv::Mat> bins(N);
    for(int i=0;i<N;++i) bins[i] = thresholdImage(raws[i]);

    cv::Mat rawL(N, raws[0].rows, CV_64F),
            rawR(N, raws[0].rows, CV_64F),
            binL(N, raws[0].rows, CV_64F),
            binR(N, raws[0].rows, CV_64F);
    for(int i=0;i<N;++i){
        for(int r=0;r<raws[i].rows;++r){
            rawL.at<double>(i,r) = raws[i].at<uchar>(r,0);
            rawR.at<double>(i,r) = raws[i].at<uchar>(r,raws[i].cols-1);
            binL.at<double>(i,r) = bins[i].at<uchar>(r,0);
            binR.at<double>(i,r) = bins[i].at<uchar>(r,bins[i].cols-1);
        }
    }

    std::vector<std::vector<double>> sim(N, std::vector<double>(N, -std::numeric_limits<double>::infinity()));
    for(int i=0;i<N;++i) for(int j=0;j<N;++j){
        if(i==j) continue;
        double s1=ssim1d(rawR.row(i), rawL.row(j));
        double s2=ssim1d(binR.row(i), binL.row(j));
        double c =crossCorr(rawR.row(i), rawL.row(j));
        sim[i][j] = alp*s1 + bet*s2 + gam*c;
    }
    // pick start = max whiteness
    std::vector<double> sumBin(N,0);
    for(int i=0;i<N;++i)
        for(int r=0;r<binL.cols;++r)
            sumBin[i] += binL.at<double>(i,r);
    int start = std::distance(sumBin.begin(), std::max_element(sumBin.begin(), sumBin.end()));

    using BeamItem = std::tuple<double, std::vector<int>, int>;
    std::vector<BeamItem> beam{{0.0, {start}, 1<<start}};

    // beam‐search
    for(int step=1;step<N;++step){
        std::vector<BeamItem> cand;
        for(auto &b: beam){
            double score; std::vector<int> path; int mask;
            std::tie(score,path,mask) = b;
            int last = path.back();
            for(int nxt=0;nxt<N;++nxt) if(!(mask&(1<<nxt))){
                cand.emplace_back(score+sim[last][nxt],
                                  [&](){auto p=path; p.push_back(nxt); return p;}(),
                                  mask|(1<<nxt));
            }
        }
        std::sort(cand.begin(), cand.end(),
                  [](auto &a, auto &b){ return std::get<0>(a)>std::get<0>(b); });
        cand.resize(std::min<int>(cand.size(), beamWidth_));
        beam.swap(cand);
    }
    // pick best full path
    auto best = *std::max_element(beam.begin(), beam.end(),
                      [](auto &a, auto &b){ return std::get<0>(a)<std::get<0>(b); });
    return std::get<1>(best);
}

std::vector<int> SurpriseManager::surprise(const std::vector<std::vector<uchar>> &bufs) const {
    std::vector<cv::Mat> imgs;
    imgs.reserve(bufs.size());
    for(auto &b:bufs){
        cv::Mat m(b), img = cv::imdecode(m, cv::IMREAD_GRAYSCALE);
        if(img.empty()) throw std::runtime_error("decode failed");
        imgs.push_back(img);
    }
    int N = imgs.size();
    if(N==1) return {0};
    if(N>=20) return beamSearch(imgs);

    // else use SSIM+DP cycle
    cv::Mat left(N, imgs[0].rows, CV_64F),
            right(N, imgs[0].rows, CV_64F);
    for(int i=0;i<N;++i){
        for(int r=0;r<imgs[i].rows;++r){
            left .at<double>(i,r) = imgs[i].at<uchar>(r,0);
            right.at<double>(i,r)= imgs[i].at<uchar>(r,imgs[i].cols-1);
        }
    }
    // append white
    cv::Mat white = cv::Mat::ones(1, imgs[0].rows, CV_64F)*255.0;
    left .push_back(white);
    right.push_back(white);

    cv::Mat ssim = metrics::calculateSSIMBatchVsBatch(right, left);

    auto [score, cycle] = findMaxWeightHamiltonianCycle(ssim, N);
    if(!cycle.empty() && cycle.front()==N) cycle.erase(cycle.begin());
    return cycle;
}

// DP‐bitmask path solver (unused)
std::pair<double,std::vector<int>>
SurpriseManager::findMaxWeightHamiltonianPath(const cv::Mat &W, int start_node) {
    int n = W.rows;
    if(n==0) return {0,{}};
    if(n==1) return {0,{0}};
    if(start_node>=n) throw std::invalid_argument("bad start");

    int FULL=1<<n; double NEG=-std::numeric_limits<double>::infinity();
    std::vector dp(FULL, std::vector<double>(n,NEG));
    std::vector parent(FULL, std::vector<int>(n,-1));
    if(start_node<0){
        for(int i=0;i<n;++i) dp[1<<i][i]=0;
    } else {
        dp[1<<start_node][start_node]=0;
    }
    for(int mask=1;mask<FULL;++mask){
        for(int j=0;j<n;++j) if(mask&(1<<j)){
            int pm=mask^(1<<j);
            if(pm==0) continue;
            for(int i=0;i<n;++i) if(pm&(1<<i)){
                double prev=dp[pm][i];
                if(prev==NEG) continue;
                double w=W.at<double>(i,j);
                if(prev+w > dp[mask][j]){
                    dp[mask][j]=prev+w;
                    parent[mask][j]=i;
                }
            }
        }
    }
    int full_mask=FULL-1; double best=NEG; int last=-1;
    for(int j=0;j<n;++j){
        if(dp[full_mask][j]>best){
            best=dp[full_mask][j]; last=j;
        }
    }
    std::vector<int> path;
    if(last>=0){
        int mask=full_mask,node=last;
        while(mask){
            path.push_back(node);
            int p=parent[mask][node];
            mask ^=1<<node;
            node=p;
            if(node<0) break;
        }
        std::reverse(path.begin(), path.end());
    }
    return {best,path};
}

std::pair<double,std::vector<int>>
SurpriseManager::findMaxWeightHamiltonianCycle(const cv::Mat &W, int start_node) {
    int n=W.rows;
    if(n<3||start_node<0||start_node>=n)
        return {-std::numeric_limits<double>::infinity(),{}};
    int FULL=1<<n; double NEG=-std::numeric_limits<double>::infinity();
    std::vector dp(FULL, std::vector<double>(n,NEG));
    std::vector parent(FULL, std::vector<int>(n,-1));
    dp[1<<start_node][start_node]=0;
    for(int mask=1;mask<FULL;++mask){
        for(int j=0;j<n;++j) if(mask&(1<<j)){
            int pm=mask^(1<<j);
            if(pm==0) continue;
            for(int i=0;i<n;++i) if(pm&(1<<i)){
                // only allow start->j when pm==(1<<start)
                if(i==start_node && pm!=(1<<start_node)) continue;
                double prev=dp[pm][i];
                if(prev==NEG) continue;
                double w=W.at<double>(i,j);
                if(prev+w>dp[mask][j]){
                    dp[mask][j]=prev+w;
                    parent[mask][j]=i;
                }
            }
        }
    }
    int full_mask=FULL-1; double best=NEG; int last=-1;
    for(int j=0;j<n;++j){
        if(j==start_node) continue;
        double v=dp[full_mask][j];
        if(v==NEG) continue;
        double tot = v + W.at<double>(j,start_node);
        if(tot>best){ best=tot; last=j; }
    }
    std::vector<int> cycle;
    if(last>=0){
        int mask=full_mask,node=last;
        while(mask){
            cycle.push_back(node);
            int p=parent[mask][node];
            mask ^=1<<node;
            node=p;
            if(node<0) break;
        }
        std::reverse(cycle.begin(), cycle.end());
    }
    return {best,cycle};
}