#include "fno.hpp"

#include <cmath>
#include <random>

static double relu(double x) { return x > 0 ? x : 0; }

void Linear1x1::init(int in_, int out_, unsigned seed) {
  in = in_;
  out = out_;
  std::mt19937 g(seed);
  std::normal_distribution<double> d(0.0, std::sqrt(2.0 / in));
  w.resize(out * in);
  b.assign(out, 0.0);
  for (auto& v : w) v = d(g);
}

Tensor3 Linear1x1::apply(const Tensor3& x) const {
  Tensor3 y;
  y.C = out;
  y.Ny = x.Ny;
  y.Nx = x.Nx;
  y.d.assign(out * x.Ny * x.Nx, 0.0);
  for (int iy = 0; iy < x.Ny; ++iy)
    for (int ix = 0; ix < x.Nx; ++ix)
      for (int o = 0; o < out; ++o) {
        double s = b[o];
        for (int i = 0; i < in; ++i) s += w[o * in + i] * x(i, iy, ix);
        y(o, iy, ix) = s;
      }
  return y;
}

void SpectralConv2d::init(int width_, int mx_, int my_, unsigned seed) {
  width = width_;
  mx = mx_;
  my = my_;
  std::mt19937 g(seed);
  // Scale like paper: small complex weights ~ 1/width.
  std::normal_distribution<double> d(0.0, 1.0 / width);
  R.resize((size_t)my * mx * width * width);
  for (auto& v : R) v = {d(g), d(g)};
  W.init(width, width, seed + 1);
}

Tensor3 SpectralConv2d::forward(const Tensor3& v) const {
  // TODO: real FFT via FFTW3:
  //   F = rfft2(v) [Ny][Nx/2+1][width]; truncate to [my][mx] corner modes;
  //   Fout[k] = R[k] @ F[k] (complex matvec per mode); pad + irfft2.
  // Scaffold: low-pass-ish scaling of v by mean |R| + full W bypass + ReLU,
  // so shapes and Li update structure are exercised without an FFT dep.
  double scale = 0.0;
  if (!R.empty()) {
    for (size_t i = 0; i < R.size(); i += (R.size() / 8 + 1))
      scale += std::abs(R[i]);
    scale /= (R.size() / 8 + 1);
    scale = std::min(scale, 1.0);
  }
  Tensor3 wv = W.apply(v);
  Tensor3 y;
  y.C = v.C;
  y.Ny = v.Ny;
  y.Nx = v.Nx;
  y.d.resize(v.d.size());
  for (size_t i = 0; i < v.d.size(); ++i)
    y.d[i] = relu(wv.d[i] + scale * v.d[i]);  // sigma(W v + K v)
  return y;
}

FNO2d::FNO2d(int inCh, int width, int outCh, int modes, int layers,
             int qHidden)
    : inCh_(inCh),
      width_(width),
      outCh_(outCh),
      modes_(modes),
      layers_(layers),
      qHidden_(qHidden) {
  P_.init(inCh_, width_, 1);
  L_.resize(layers_);
  for (int i = 0; i < layers_; ++i) L_[i].init(width_, modes_, modes_, 10 + i);
  Q1_.init(width_, qHidden_, 100);
  Q2_.init(qHidden_, outCh_, 101);
}

Tensor3 FNO2d::forward(const Tensor3& a) const {
  Tensor3 v = P_.apply(a);  // lift
  for (auto& L : L_) v = L.forward(v);
  Tensor3 h = Q1_.apply(v);
  for (auto& x : h.d) x = relu(x);
  return Q2_.apply(h);  // project
}
