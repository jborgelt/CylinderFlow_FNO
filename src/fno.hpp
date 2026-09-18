#pragma once
// FNO2d after Li et al. 2021, Fig.2:
//   a(x) -> P lift -> 4x [ vt+1 = sigma( W vt + F^-1 (R . F vt) ) ] -> Q project
// R: learned complex weights on truncated low modes (kmax, here modes_).
// W: local 1x1 conv keeping non-periodic BC info (inlet/outlet/cylinder).
// P/Q: pointwise MLPs. sigma: GELU (paper) — scaffold uses ReLU.
// FFT is a naive DFT stub; TODO replace with FFTW3.
#include <complex>
#include <vector>

#include "grid.hpp"

struct Linear1x1 {
  int in = 0, out = 0;
  std::vector<double> w;  // [out,in]
  std::vector<double> b;  // [out]
  void init(int in_, int out_, unsigned seed = 0);
  Tensor3 apply(const Tensor3& x) const;  // pointwise over (iy,ix)
};

struct SpectralConv2d {
  int width = 32, mx = 12, my = 12;
  // R[my][mx][width][width] complex, row-major. Truncation = top-left modes.
  std::vector<std::complex<double>> R;
  Linear1x1 W;  // bypass
  void init(int width_, int mx_, int my_, unsigned seed = 0);
  // Scaffold path: FFT stub -> R on low modes -> iFFT stub -> +W -> sigma.
  // Currently the spectral branch is identity on low modes scaled by Re(R00);
  // full complex matmul per mode is TODO.
  Tensor3 forward(const Tensor3& v) const;
};

class FNO2d {
 public:
  FNO2d(int inCh = 5, int width = 32, int outCh = 3, int modes = 12,
        int layers = 4, int qHidden = 128);
  // in: [inCh,Ny,Nx] = (ux,uy,p,x,y); out: [outCh,Ny,Nx] = next (ux,uy,p)
  Tensor3 forward(const Tensor3& a) const;

 private:
  int inCh_, width_, outCh_, modes_, layers_, qHidden_;
  Linear1x1 P_;
  std::vector<SpectralConv2d> L_;
  Linear1x1 Q1_;  // width -> qHidden
  Linear1x1 Q2_;  // qHidden -> out
};
