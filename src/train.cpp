#include "train.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

void Normalizer::fit(const std::vector<Sample>& data) {
  if (data.empty()) return;
  const int C = data[0].x.C;
  mean_.assign(C, 0.0);
  std_.assign(C, 1.0);
  double n = 0;
  for (auto& s : data) {
    // per-channel mean over all cells/samples
    for (int c = 0; c < C; ++c) {
      double sum = 0;
      const size_t cells = (size_t)s.x.Ny * s.x.Nx;
      for (size_t i = 0; i < cells; ++i) sum += s.x.d[(c * s.x.Ny) * s.x.Nx + i];
      mean_[c] += sum;
    }
    n += (double)s.x.Ny * s.x.Nx;
  }
  for (int c = 0; c < C; ++c) mean_[c] /= (n * 1.0);
  // std
  double tot = 0;
  for (auto& s : data) {
    const size_t cells = (size_t)s.x.Ny * s.x.Nx;
    for (int c = 0; c < C; ++c)
      for (size_t i = 0; i < cells; ++i) {
        double d = s.x.d[(c * s.x.Ny) * s.x.Nx + i] - mean_[c];
        std_[c] += d * d;
      }
    tot += (double)cells;
  }
  for (int c = 0; c < C; ++c) {
    std_[c] = std::sqrt(std_[c] / tot);
    if (std_[c] < 1e-8) std_[c] = 1.0;
  }
}

Tensor3 Normalizer::encode(const Tensor3& x) const {
  Tensor3 y = x;
  for (int c = 0; c < x.C; ++c)
    for (int iy = 0; iy < x.Ny; ++iy)
      for (int ix = 0; ix < x.Nx; ++ix)
        y(c, iy, ix) = (x(c, iy, ix) - mean_[c]) / std_[c];
  return y;
}

Tensor3 Normalizer::decode(const Tensor3& xn) const {
  // Only valid if xn.C == mean_.size(); y-targets stay unnormalized for now.
  Tensor3 y = xn;
  const int C = std::min<int>(xn.C, mean_.size());
  for (int c = 0; c < C; ++c)
    for (int iy = 0; iy < xn.Ny; ++iy)
      for (int ix = 0; ix < xn.Nx; ++ix)
        y(c, iy, ix) = xn(c, iy, ix) * std_[c] + mean_[c];
  return y;
}

std::vector<Sample> buildDataset(FoamReader& reader, UniformGrid& grid,
                                 const CellCenters& centers,
                                 const std::vector<double>& times) {
  std::vector<Sample> out;
  if (times.size() < 2) return out;
  Tensor3 coords = grid.coords();
  for (size_t k = 0; k + 1 < times.size(); ++k) {
    Snapshot sx = reader.readTime(times[k]);
    Snapshot sy = reader.readTime(times[k + 1]);
    // t=0 holds uniform ICs (single value), not per-cell fields -> no pair.
    if ((size_t)sx.nCells != centers.x.size() ||
        (size_t)sy.nCells != centers.x.size()) {
      std::printf("skip pair %.0f->%.0f (uniform IC, no per-cell data)\n",
                  times[k], times[k + 1]);
      continue;
    }
    Tensor3 gx = grid.createData(sx, centers);
    Tensor3 gy = grid.createData(sy, centers);
    Sample s;
    s.tIn = times[k];
    s.tOut = times[k + 1];
    s.x.C = 5;
    s.x.Ny = gx.Ny;
    s.x.Nx = gx.Nx;
    s.x.d.resize((size_t)5 * gx.Ny * gx.Nx);
    for (int iy = 0; iy < gx.Ny; ++iy)
      for (int ix = 0; ix < gx.Nx; ++ix) {
        s.x(0, iy, ix) = gx(0, iy, ix);
        s.x(1, iy, ix) = gx(1, iy, ix);
        s.x(2, iy, ix) = gx(2, iy, ix);
        s.x(3, iy, ix) = coords(0, iy, ix);
        s.x(4, iy, ix) = coords(1, iy, ix);
      }
    s.y = gy;
    out.push_back(std::move(s));
  }
  return out;
}

static void tensorStats(const Tensor3& t, double& mn, double& mx, double& mean,
                        bool& hasNan, int ch) {
  mn = std::numeric_limits<double>::infinity();
  mx = -std::numeric_limits<double>::infinity();
  double sum = 0;
  size_t n = 0;
  hasNan = false;
  for (int iy = 0; iy < t.Ny; ++iy)
    for (int ix = 0; ix < t.Nx; ++ix) {
      double v = t(ch, iy, ix);
      if (std::isnan(v) || std::isinf(v)) {
        hasNan = true;
        continue;
      }
      mn = std::min(mn, v);
      mx = std::max(mx, v);
      sum += v;
      ++n;
    }
  mean = n ? sum / n : 0;
}

void debugPrintTensor(const char* name, const Tensor3& t, int maxVals) {
  std::printf("%s: C=%d Ny=%d Nx=%d\n", name, t.C, t.Ny, t.Nx);
  for (int c = 0; c < t.C; ++c) {
    double mn, mx, mean;
    bool hasNan;
    tensorStats(t, mn, mx, mean, hasNan, c);
    std::printf("  ch%d: min=%.5f max=%.5f mean=%.5f%s first=[", c, mn, mx,
                mean, hasNan ? " HAS_NAN/INF!" : "");
    int shown = 0;
    for (int iy = 0; iy < t.Ny && shown < maxVals; ++iy)
      for (int ix = 0; ix < t.Nx && shown < maxVals; ++ix, ++shown)
        std::printf("%s%.4f", shown ? "," : "", t(c, iy, ix));
    std::printf("]\n");
  }
}

void debugPrintSample(const Sample& s, int idx) {
  std::printf("--- sample %d: t %.2f -> %.2f ---\n", idx, s.tIn, s.tOut);
  debugPrintTensor("x", s.x);
  debugPrintTensor("y", s.y);
}

void inspectDataset(const std::vector<Sample>& data) {
  std::printf("dataset: %zu next-step pairs\n", data.size());
  for (size_t i = 0; i < data.size(); ++i) debugPrintSample(data[i], (int)i);
}

Trainer::Trainer(FNO2d& model, double lr) : model_(model), lr_(lr) {}

double Trainer::loss(const Tensor3& pred, const Tensor3& y) const {
  double s = 0;
  const size_t n = std::min(pred.d.size(), y.d.size());
  for (size_t i = 0; i < n; ++i) {
    double d = pred.d[i] - y.d[i];
    s += d * d;
  }
  return n ? s / n : 0;
}

double Trainer::trainStep(const Sample& s) {
  // TODO: backward() through Q/L/P + Adam update of R,W,P,Q.
  // Scaffold: forward + loss so data path is verified end to end.
  Tensor3 xn = normFitted_ ? norm_.encode(s.x) : s.x;
  Tensor3 pred = model_.forward(xn);
  return loss(pred, s.y);
}

void Trainer::fit(std::vector<Sample> data, int epochs) {
  norm_.fit(data);
  normFitted_ = true;
  for (int e = 0; e < epochs; ++e) {
    double tot = 0;
    for (auto& s : data) tot += trainStep(s);
    std::printf("[train] epoch %d/%d  mse=%.6f  (no weight update yet)\n", e + 1,
                epochs, tot / (data.empty() ? 1 : data.size()));
  }
}
