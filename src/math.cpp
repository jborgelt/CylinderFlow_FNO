#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <fftw3.h>
#include <stdexcept>

namespace math {

double relu(double x) { return x > 0 ? x : 0; }

double gelu(double x) {
  static const double kSqrt2 = 1.4142135623730951;
  return 0.5 * x * (1.0 + std::erf(x / kSqrt2));
}

double geluDeriv(double x) {
  // d/dx [0.5x(1+erf(x/sqrt2))] = 0.5(1+erf(x/sqrt2)) + x*phi(x),
  // phi = standard normal pdf.
  static const double kSqrt2 = 1.4142135623730951;
  static const double kInvSqrt2Pi = 0.3989422804014327;
  const double phi = kInvSqrt2Pi * std::exp(-0.5 * x * x);
  return 0.5 * (1.0 + std::erf(x / kSqrt2)) + x * phi;
}

double reluDeriv(double x) { return x > 0 ? 1.0 : 0.0; }

void heInit(std::vector<double> &m, int rows, int cols, std::mt19937 &gen) {
  std::normal_distribution<double> d(0.0, std::sqrt(2.0 / cols));
  m.resize((size_t)rows * cols);
  for (auto &v : m) v = d(gen);
}

void spectralInit(std::vector<std::complex<double>> &m, int width,
                  std::mt19937 &gen) {
  std::normal_distribution<double> d(0.0, 1.0 / width);
  for (auto &v : m) v = {d(gen), d(gen)};
}

void matvecAdd(const std::vector<double> &W, const std::vector<double> &b,
               const double *x, double *y, int in, int out) {
  for (int o = 0; o < out; ++o) {
    double s = b[o];
    const double *row = &W[(size_t)o * in];
    for (int i = 0; i < in; ++i) s += row[i] * x[i];
    y[o] = s;
  }
}

void spectralMatvec(const std::vector<std::complex<double>> &R,
                    const std::complex<double> *in, std::complex<double> *out,
                    int nModes, int width) {
  for (int m = 0; m < nModes; ++m) {
    const std::complex<double> *Rm = &R[(size_t)m * width * width];
    const std::complex<double> *xm = in + (size_t)m * width;
    std::complex<double> *ym = out + (size_t)m * width;
    for (int o = 0; o < width; ++o) {
      std::complex<double> s(0, 0);
      for (int i = 0; i < width; ++i) s += Rm[(size_t)o * width + i] * xm[i];
      ym[o] = s;
    }
  }
}

struct FFT2D::Impl {
  fftw_plan fwd = nullptr, bwd = nullptr;
  double *real = nullptr;
  fftw_complex *cplx = nullptr;
  int nx = 0, ny = 0, nc = 0;
};

FFT2D::FFT2D(int nx, int ny) : nx_(nx), ny_(ny), p_(new Impl) {
  if (nx <= 0 || ny <= 0) throw std::runtime_error("FFT2D: bad size");
  p_->nx = nx;
  p_->ny = ny;
  p_->nc = ny * (nx / 2 + 1);
  p_->real = (double *)fftw_malloc(sizeof(double) * (size_t)nx * ny);
  p_->cplx = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * p_->nc);
  if (!p_->real || !p_->cplx) throw std::runtime_error("FFT2D: alloc failed");
  p_->fwd = fftw_plan_dft_r2c_2d(ny, nx, p_->real, p_->cplx, FFTW_MEASURE);
  p_->bwd = fftw_plan_dft_c2r_2d(ny, nx, p_->cplx, p_->real, FFTW_MEASURE);
  if (!p_->fwd || !p_->bwd) throw std::runtime_error("FFT2D: plan failed");
}

FFT2D::~FFT2D() {
  if (!p_) return;
  fftw_destroy_plan(p_->fwd);
  fftw_destroy_plan(p_->bwd);
  fftw_free(p_->real);
  fftw_free(p_->cplx);
  delete p_;
}

std::vector<std::complex<double>> FFT2D::forward(
    const std::vector<double> &in) {
  if (in.size() != (size_t)nx_ * ny_)
    throw std::runtime_error("FFT2D::forward: size mismatch");
  std::copy(in.begin(), in.end(), p_->real);
  fftw_execute(p_->fwd);
  std::vector<std::complex<double>> out(p_->nc);
  for (int i = 0; i < p_->nc; ++i) out[i] = {p_->cplx[i][0], p_->cplx[i][1]};
  return out;
}

std::vector<double> FFT2D::backward(
    const std::vector<std::complex<double>> &in) {
  if ((int)in.size() != p_->nc)
    throw std::runtime_error("FFT2D::backward: size mismatch");
  for (int i = 0; i < p_->nc; ++i) {
    p_->cplx[i][0] = in[i].real();
    p_->cplx[i][1] = in[i].imag();
  }
  fftw_execute(p_->bwd);
  const double scale = 1.0 / ((size_t)nx_ * ny_);
  std::vector<double> out((size_t)nx_ * ny_);
  for (size_t i = 0; i < out.size(); ++i) out[i] = p_->real[i] * scale;
  return out;
}

double mse(const std::vector<double> &pred,
           const std::vector<double> &target) {
  const size_t n = std::min(pred.size(), target.size());
  if (n == 0) return 0;
  double s = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = pred[i] - target[i];
    s += d * d;
  }
  return s / n;
}

double relativeL2(const std::vector<double> &pred,
                  const std::vector<double> &target) {
  const size_t n = std::min(pred.size(), target.size());
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = pred[i] - target[i];
    num += d * d;
    den += target[i] * target[i];
  }
  if (den <= 0) return std::sqrt(num);
  return std::sqrt(num / den);
}

void AdamState::init(size_t n) {
  m.assign(n, 0.0);
  v.assign(n, 0.0);
  t = 0;
}

void AdamState::step(std::vector<double> &p,
                     const std::vector<double> &grad, double lr, double beta1,
                     double beta2, double eps) {
  if (m.size() != p.size()) init(p.size());
  ++t;
  const double b1t = 1.0 - std::pow(beta1, t);
  const double b2t = 1.0 - std::pow(beta2, t);
  for (size_t i = 0; i < p.size(); ++i) {
    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
    p[i] -= lr * (m[i] / b1t) / (std::sqrt(v[i] / b2t) + eps);
  }
}

bool hasNonFinite(const std::vector<double> &v) {
  for (double x : v)
    if (!std::isfinite(x)) return true;
  return false;
}

}  // namespace math
