#pragma once
// FNO2d after Li et al. 2021, Fig.2:
//   a(x) -> P lift -> 4x [ vt+1 = sigma( W vt + F^-1 (R . F vt) ) ] -> Q project
// R: learned complex weights on truncated low modes (kmax, here modes_).
// W: local 1x1 conv keeping non-periodic BC info (inlet/outlet/cylinder).
// P/Q: pointwise MLPs. sigma: GELU (paper) — scaffold uses ReLU.
// FFT is a naive DFT stub; TODO replace with FFTW3.
#include <complex>
#include <string>
#include <vector>

#include "grid.hpp"
#include "weights.hpp"

struct Linear1x1 {
  int in = 0, out = 0;
  std::vector<double> w;  // [out,in]
  std::vector<double> b;  // [out]
  void init(int in_, int out_, unsigned seed = 0);
  // Truncated/padded identity (diag 1 over min(in,out)), bias 0.
  void setIdentity();
  Tensor3 apply(const Tensor3& x) const;  // pointwise over (iy,ix)
  // Append "<prefix>.weight [out,in]" + "<prefix>.bias [out]".
  void collectWeights(const std::string& prefix,
                      std::vector<WeightTensor>& dst) const;
  void applyWeights(const WeightTensor& wgt, const WeightTensor& bias);
};

struct SpectralConv2d {
  int width = 32, mx = 12, my = 12;
  // R[my][mx][width][width] complex, row-major. Truncation = top-left modes.
  std::vector<std::complex<double>> R;
  Linear1x1 W;  // bypass
  void init(int width_, int mx_, int my_, unsigned seed = 0);
  void setIdentity();  // R = 0, W = identity: layer ~= GELU(v)
  // Scaffold path: FFT stub -> R on low modes -> iFFT stub -> +W -> sigma.
  // Currently the spectral branch is identity on low modes scaled by Re(R00);
  // full complex matmul per mode is TODO.
  Tensor3 forward(const Tensor3& v) const;
};

class FNO2d {
 public:
  FNO2d(int inCh = 5, int width = 32, int outCh = 3, int modes = 12,
        int layers = 4, int qHidden = 128, int pLayers = 1, int qLayers = 2);
  // in: [inCh,Ny,Nx] = (ux,uy,p,x,y); out: [outCh,Ny,Nx] = next (ux,uy,p)
  Tensor3 forward(const Tensor3& a) const;
  // Identity start: R = 0, all linears identity. forward(x) ~= x up to the
  // GELU activations (exact identity is impossible with GELU in the path;
  // GELU(x) = x only asymptotically for x >> 0, and 0 for x << 0).
  void initIdentity();
  // Weight file IO (safetensors-compatible, see weights.hpp). Load verifies
  // the stored arch matches this model and throws otherwise.
  void save(const std::string& path, bool fp32 = false) const;
  void load(const std::string& path);

 private:
  int inCh_, width_, outCh_, modes_, layers_, qHidden_;
  int pLayers_, qLayers_;
  // Lift stack P: layer 0 is inCh->width, the rest width->width.
  // Projection stack Q: width->qHidden -> ... -> qHidden->outCh
  // (single width->outCh layer when qLayers == 1).
  std::vector<Linear1x1> P_, Q_;
  std::vector<SpectralConv2d> L_;
};

// Rollout feedback (in-memory only, never via files): build the next model
// input from a previous output. First 3 output channels become next
// (ux,uy,p); coords are reattached fresh; extra channels zero-padded,
// missing coords truncated -- same rules as the initial input builder.
Tensor3 feedbackInput(const Tensor3& y, const UniformGrid& grid, int inCh);
