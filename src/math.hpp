#pragma once
// math: all shared numerics for the from-scratch FNO.
// Activations, weight init, small linear-algebra helpers, an FFTW-backed
// real 2D FFT, per-mode complex spectral matmuls, losses and Adam.
// Everything operates on std::vector (row-major); no Eigen dependency.
#include <complex>
#include <random>
#include <vector>

namespace math {

// --- Activations (Li et al. uses GELU; relu kept for debugging) ---
double relu(double x);
double gelu(double x);        // exact: 0.5*x*(1+erf(x/sqrt(2)))
double geluDeriv(double x);   // for the future backward pass
double reluDeriv(double x);

// --- Weight init ---
// He init for real [rows x cols] matrices (good with ReLU/GELU).
void heInit(std::vector<double> &m, int rows, int cols, std::mt19937 &gen);
// Small complex weights ~ N(0, 1/width) per component, as in the paper's R.
void spectralInit(std::vector<std::complex<double>> &m, int width,
                  std::mt19937 &gen);

// --- Tiny linear algebra (row-major) ---
// y = W x + b, W [out x in]. Applied pointwise by callers.
void matvecAdd(const std::vector<double> &W, const std::vector<double> &b,
               const double *x, double *y, int in, int out);

// Per-mode complex transform: for each of nModes, out[m] = R[m] @ in[m],
// R[m] is [width x width] complex, in[m]/out[m] length width.
void spectralMatvec(const std::vector<std::complex<double>> &R,
                    const std::complex<double> *in, std::complex<double> *out,
                    int nModes, int width);

// --- Real 2D FFT via FFTW3 ---
// Forward:  [Ny x Nx] real -> [Ny x (Nx/2+1)] complex (r2c).
// Backward: complex -> real, normalized by 1/(Nx*Ny) like the paper's F^-1.
class FFT2D {
 public:
  FFT2D(int nx, int ny);
  ~FFT2D();
  FFT2D(const FFT2D &) = delete;
  FFT2D &operator=(const FFT2D &) = delete;
  std::vector<std::complex<double>> forward(const std::vector<double> &in);
  std::vector<double> backward(const std::vector<std::complex<double>> &in);
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nComplex() const { return ny_ * (nx_ / 2 + 1); }

 private:
  int nx_, ny_;
  struct Impl;
  Impl *p_;
};

// --- Losses ---
double mse(const std::vector<double> &pred, const std::vector<double> &target);
// Relative L2 from Li et al.: ||pred-true||_2 / ||true||_2 (training metric).
double relativeL2(const std::vector<double> &pred,
                  const std::vector<double> &target);

// --- Adam (for the future weight update; state per parameter tensor) ---
struct AdamState {
  std::vector<double> m, v;
  int t = 0;
  void init(size_t n);
  // Single param step: p -= lr * m^/(sqrt(v^)+eps).
  void step(std::vector<double> &p, const std::vector<double> &grad,
            double lr, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-8);
};

// --- Sanity ---
bool hasNonFinite(const std::vector<double> &v);

}  // namespace math
