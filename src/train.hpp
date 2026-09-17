#pragma once
// Trainer: next-snapshot fit X(t) -> Y(t+dt) with MSE.
// Paper (Li Sec.5.3) reports relative L2; scaffold tracks plain MSE.
// No autograd yet -> trainStep() only evaluates loss (TODO: backward+Adam).
#include <vector>

#include "fno.hpp"
#include "grid.hpp"

struct Sample {
  double tIn = 0.0, tOut = 0.0;  // snapshot times: y is one step after x
  Tensor3 x;                     // [inCh,Ny,Nx]
  Tensor3 y;                     // [3,Ny,Nx] next snapshot
};

// Build next-snapshot pairs X(t)->Y(t+dt) on the uniform grid.
// x gets (ux,uy,p,x,y) channels, y gets (ux,uy,p).
std::vector<Sample> buildDataset(FoamReader& reader, UniformGrid& grid,
                                 const CellCenters& centers,
                                 const std::vector<double>& times);

// Debug/inspect helpers: cout shapes, per-channel stats, NaN checks.
void debugPrintTensor(const char* name, const Tensor3& t, int maxVals = 5);
void debugPrintSample(const Sample& s, int idx);
void inspectDataset(const std::vector<Sample>& data);

class Normalizer {
 public:
  void fit(const std::vector<Sample>& data);  // per-channel mean/std of x
  Tensor3 encode(const Tensor3& x) const;
  Tensor3 decode(const Tensor3& xn) const;

 private:
  std::vector<double> mean_, std_;
};

class Trainer {
 public:
  Trainer(FNO2d& model, double lr = 1e-3);
  double loss(const Tensor3& pred, const Tensor3& y) const;  // MSE
  double trainStep(const Sample& s);  // scaffold: forward+loss only
  void fit(std::vector<Sample> data, int epochs);

 private:
  FNO2d& model_;
  double lr_;
  Normalizer norm_;
  bool normFitted_ = false;
};
