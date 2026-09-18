#include "fno.hpp"
#include "math.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <stdexcept>

void Linear1x1::init(int in_, int out_, unsigned seed) {
  in = in_;
  out = out_;
  std::mt19937 g(seed);
  math::heInit(w, out, in, g);
  b.assign(out, 0.0);
}

void Linear1x1::setIdentity() {
  w.assign((size_t)out * in, 0.0);
  for (int o = 0; o < out && o < in; ++o) w[(size_t)o * in + o] = 1.0;
  b.assign(out, 0.0);
}

void Linear1x1::collectWeights(const std::string& prefix,
                               std::vector<WeightTensor>& dst) const {
  dst.push_back({prefix + ".weight", {(int64_t)out, (int64_t)in}, w});
  dst.push_back({prefix + ".bias", {(int64_t)out}, b});
}

void Linear1x1::applyWeights(const WeightTensor& wgt,
                             const WeightTensor& bias) {
  if (wgt.shape != std::vector<int64_t>{(int64_t)out, (int64_t)in} ||
      wgt.data.size() != w.size())
    throw std::runtime_error("weights: shape mismatch for '" + wgt.name +
                             "'");
  if (bias.shape != std::vector<int64_t>{(int64_t)out} ||
      bias.data.size() != b.size())
    throw std::runtime_error("weights: shape mismatch for '" + bias.name +
                             "'");
  w = wgt.data;
  b = bias.data;
}

Tensor3 Linear1x1::apply(const Tensor3& x) const {
  Tensor3 y;
  y.C = out;
  y.Ny = x.Ny;
  y.Nx = x.Nx;
  y.d.assign((size_t)out * x.Ny * x.Nx, 0.0);
  // Tensor3 is channel-first, so per-pixel channels are strided; gather to
  // a contiguous buffer and use the shared math kernel (bit-identical order).
  std::vector<double> xin(in), yout(out);
  for (int iy = 0; iy < x.Ny; ++iy)
    for (int ix = 0; ix < x.Nx; ++ix) {
      for (int i = 0; i < in; ++i) xin[i] = x(i, iy, ix);
      math::matvecAdd(w, b, xin.data(), yout.data(), in, out);
      for (int o = 0; o < out; ++o) y(o, iy, ix) = yout[o];
    }
  return y;
}

void SpectralConv2d::init(int width_, int mx_, int my_, unsigned seed) {
  width = width_;
  mx = mx_;
  my = my_;
  std::mt19937 g(seed);
  R.resize((size_t)my * mx * width * width);
  math::spectralInit(R, width, g);  // small complex weights ~ 1/width
  W.init(width, width, seed + 1);
}

void SpectralConv2d::setIdentity() {
  R.assign(R.size(), {0.0, 0.0});
  W.setIdentity();
}

Tensor3 SpectralConv2d::forward(const Tensor3& v) const {
  // TODO: real spectral path via math::FFT2D + math::spectralMatvec:
  //   F = fft.forward(channel); truncate to [my][mx] corner modes;
  //   Fout = R @ F per mode; pad + fft.backward. Scaffold below keeps the
  //   Li update structure sigma(W v + K v) with a stubbed K.
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
    y.d[i] = math::gelu(wv.d[i] + scale * v.d[i]);  // sigma(W v + K v)
  return y;
}

FNO2d::FNO2d(int inCh, int width, int outCh, int modes, int layers,
             int qHidden, int pLayers, int qLayers)
    : inCh_(inCh),
      width_(width),
      outCh_(outCh),
      modes_(modes),
      layers_(layers),
      qHidden_(qHidden),
      pLayers_(pLayers),
      qLayers_(qLayers) {
  if (pLayers_ < 1 || qLayers_ < 1)
    throw std::runtime_error("FNO2d: pLayers and qLayers must be >= 1");
  P_.resize(pLayers_);
  P_[0].init(inCh_, width_, 1);
  for (int i = 1; i < pLayers_; ++i) P_[i].init(width_, width_, 1 + i);
  L_.resize(layers_);
  for (int i = 0; i < layers_; ++i) L_[i].init(width_, modes_, modes_, 10 + i);
  Q_.resize(qLayers_);
  for (int i = 0; i < qLayers_; ++i) {
    const int in = (i == 0) ? width_ : qHidden_;
    const int out = (i + 1 == qLayers_) ? outCh_ : qHidden_;
    Q_[i].init(in, out, 100 + i);
  }
}

static void geluInplace(Tensor3& t) {
  for (auto& x : t.d) x = math::gelu(x);
}

Tensor3 FNO2d::forward(const Tensor3& a) const {
  Tensor3 v = a;
  
  // Uplifting
  for (size_t i = 0; i < P_.size(); ++i) {  // lift stack
    v = P_[i].apply(v);
    if (i + 1 < P_.size()) geluInplace(v);
  }
  
  // FNO layer
  for (auto &L : L_)
    v = L.forward(v);
  
  // Projection 
  Tensor3 h = v;
  for (size_t i = 0; i < Q_.size(); ++i) {  // projection stack
    h = Q_[i].apply(h);
    if (i + 1 < Q_.size()) geluInplace(h);
  }
  return h;
}

void FNO2d::initIdentity() {
  for (auto& P : P_) P.setIdentity();
  for (auto& L : L_) L.setIdentity();
  for (auto& Q : Q_) Q.setIdentity();
}

namespace {
// R is [my,mx,width,width] complex; stored as R_real/R_imag with same shape.
void collectR(const std::string& prefix,
              const std::vector<std::complex<double>>& R, int my, int mx,
              int width, std::vector<WeightTensor>& dst) {
  WeightTensor re{prefix + ".R_real",
                  {(int64_t)my, (int64_t)mx, (int64_t)width, (int64_t)width},
                  {}},
              im = re;
  im.name = prefix + ".R_imag";
  re.data.reserve(R.size());
  im.data.reserve(R.size());
  for (auto& v : R) {
    re.data.push_back(v.real());
    im.data.push_back(v.imag());
  }
  dst.push_back(std::move(re));
  dst.push_back(std::move(im));
}

std::vector<std::complex<double>> applyR(
    const WeightTensor& re, const WeightTensor& im, int my, int mx,
    int width) {
  const std::vector<int64_t> want{(int64_t)my, (int64_t)mx, (int64_t)width,
                                  (int64_t)width};
  if (re.shape != want || im.shape != want ||
      re.data.size() != im.data.size() ||
      re.data.size() != (size_t)my * mx * width * width)
    throw std::runtime_error("weights: R shape mismatch for '" + re.name +
                             "'");
  std::vector<std::complex<double>> R(re.data.size());
  for (size_t i = 0; i < R.size(); ++i) R[i] = {re.data[i], im.data[i]};
  return R;
}

const WeightTensor& findTensor(const std::vector<WeightTensor>& ts,
                               const std::string& name) {
  for (auto& t : ts)
    if (t.name == name) return t;
  throw std::runtime_error("weights: missing tensor '" + name + "'");
}

std::map<std::string, std::string> archMeta(int inCh, int width, int outCh,
                                            int modes, int layers,
                                            int qHidden, int pLayers,
                                            int qLayers) {
  return {{"format", "cylfno-safetensors/1"},
          {"inCh", std::to_string(inCh)},
          {"width", std::to_string(width)},
          {"outCh", std::to_string(outCh)},
          {"modes", std::to_string(modes)},
          {"layers", std::to_string(layers)},
          {"qHidden", std::to_string(qHidden)},
          {"pLayers", std::to_string(pLayers)},
          {"qLayers", std::to_string(qLayers)}};
}

void checkMeta(const std::map<std::string, std::string>& meta,
               const char* key, int want) {
  auto it = meta.find(key);
  if (it == meta.end() || it->second != std::to_string(want))
    throw std::runtime_error(std::string("weights: arch mismatch for '") +
                             key + "' (file has '" +
                             (it == meta.end() ? "<missing>" : it->second) +
                             "')");
}
}  // namespace

void FNO2d::save(const std::string& path, bool fp32) const {
  std::vector<WeightTensor> ts;
  for (size_t i = 0; i < P_.size(); ++i)
    P_[i].collectWeights("P." + std::to_string(i), ts);
  for (size_t k = 0; k < L_.size(); ++k) {
    const std::string p = "layers." + std::to_string(k);
    collectR(p, L_[k].R, L_[k].my, L_[k].mx, L_[k].width, ts);
    L_[k].W.collectWeights(p + ".W", ts);
  }
  for (size_t i = 0; i < Q_.size(); ++i)
    Q_[i].collectWeights("Q." + std::to_string(i), ts);
  saveWeights(path, ts,
              archMeta(inCh_, width_, outCh_, modes_, (int)L_.size(),
                       qHidden_, (int)P_.size(), (int)Q_.size()),
              fp32);
}

void FNO2d::load(const std::string& path) {
  auto [ts, meta] = loadWeights(path);
  auto fmt = meta.find("format");
  if (fmt == meta.end() || fmt->second != "cylfno-safetensors/1")
    throw std::runtime_error("weights: not a cylfno model file");
  checkMeta(meta, "inCh", inCh_);
  checkMeta(meta, "width", width_);
  checkMeta(meta, "outCh", outCh_);
  checkMeta(meta, "modes", modes_);
  checkMeta(meta, "layers", (int)L_.size());
  checkMeta(meta, "qHidden", qHidden_);
  checkMeta(meta, "pLayers", (int)P_.size());
  checkMeta(meta, "qLayers", (int)Q_.size());
  for (size_t i = 0; i < P_.size(); ++i) {
    const std::string p = "P." + std::to_string(i);
    P_[i].applyWeights(findTensor(ts, p + ".weight"), findTensor(ts, p + ".bias"));
  }
  for (size_t k = 0; k < L_.size(); ++k) {
    const std::string p = "layers." + std::to_string(k);
    L_[k].R = applyR(findTensor(ts, p + ".R_real"),
                     findTensor(ts, p + ".R_imag"), L_[k].my, L_[k].mx,
                     L_[k].width);
    L_[k].W.applyWeights(findTensor(ts, p + ".W.weight"),
                         findTensor(ts, p + ".W.bias"));
  }
  for (size_t i = 0; i < Q_.size(); ++i) {
    const std::string p = "Q." + std::to_string(i);
    Q_[i].applyWeights(findTensor(ts, p + ".weight"), findTensor(ts, p + ".bias"));
  }
}

Tensor3 feedbackInput(const Tensor3& y, const UniformGrid& grid, int inCh) {
  if (y.C < 3) throw std::runtime_error("feedbackInput: need >= 3 channels");
  if (inCh < 3) throw std::runtime_error("feedbackInput: inCh must be >= 3");
  Tensor3 c = grid.coords();
  Tensor3 x;
  x.C = inCh;
  x.Ny = y.Ny;
  x.Nx = y.Nx;
  x.d.assign((size_t)inCh * y.Ny * y.Nx, 0.0);
  for (int iy = 0; iy < y.Ny; ++iy)
    for (int ix = 0; ix < y.Nx; ++ix) {
      x(0, iy, ix) = y(0, iy, ix);
      x(1, iy, ix) = y(1, iy, ix);
      x(2, iy, ix) = y(2, iy, ix);
      if (inCh > 3) x(3, iy, ix) = c(0, iy, ix);
      if (inCh > 4) x(4, iy, ix) = c(1, iy, ix);
    }
  return x;
}
