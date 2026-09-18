// Driver:
//   ./cylfno [caseDir] [time]  -> predict: one input state -> prediction + file
//   ./cylfno inspect [caseDir] -> load ALL snapshots, build next-step pairs,
//                                 debug-print the data structures
//   ./cylfno probe [caseDir] [seed SEED] t1 t2 ... -> one random grid cell
//                                 printed across the given timesteps
//   ./cylfno save <file> [fp32] -> random-init model from TOML, save weights
//   ./cylfno wtest <file> [fp32] -> save->load roundtrip, require identical out
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

#include "config.hpp"
#include "fno.hpp"
#include "foam_reader.hpp"
#include "grid.hpp"
#include "vtk.hpp"
#include "weights.hpp"

static void stats(const char *name, const std::vector<double> &v);

// Build [inCh,Ny,Nx] model input from created grid data + coords.
static Tensor3 buildModelInput(const Tensor3 &g, const UniformGrid &grid,
                               int inCh) {
  if (inCh < 3) throw std::runtime_error("fno_channelIn must be >= 3");
  Tensor3 c = grid.coords();
  Tensor3 x;
  x.C = inCh;
  x.Ny = g.Ny;
  x.Nx = g.Nx;
  x.d.assign((size_t)inCh * g.Ny * g.Nx, 0.0);
  for (int iy = 0; iy < g.Ny; ++iy)
    for (int ix = 0; ix < g.Nx; ++ix) {
      x(0, iy, ix) = g(0, iy, ix);
      x(1, iy, ix) = g(1, iy, ix);
      x(2, iy, ix) = g(2, iy, ix);
      if (inCh > 3) x(3, iy, ix) = c(0, iy, ix);
      if (inCh > 4) x(4, iy, ix) = c(1, iy, ix);
    }
  return x;
}

static FNO2d makeModel(const Config &cfg) {
  FNO2d model(cfg.fno_channelIn, cfg.fno_width, cfg.fno_channelOut,
              cfg.fno_modes, cfg.fno_layers, cfg.q_hidden, cfg.p_layers,
              cfg.q_layers);
  // Identity start (output ~= input up to GELU), then try the TOML path.
  model.initIdentity();
  if (!cfg.weight_file.empty()) {
    std::ifstream wf(cfg.weight_file, std::ios::binary);
    if (wf) {
      model.load(cfg.weight_file);
      std::printf("loaded weights from %s\n", cfg.weight_file.c_str());
    } else {
      std::printf("note: weight_file '%s' not found, keeping identity init\n",
                  cfg.weight_file.c_str());

      printf("Save newly initialized model weights %s", cfg.weight_file.c_str());
      std::filesystem::path p(cfg.weight_file);
      if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
      model.save(cfg.weight_file);
    }
  }
  return model;
}

// Write predicted field via the weights engine (readable by safedump.py).
static void writePrediction(const std::string &path, const Tensor3 &y,
                            double t) {
  std::vector<WeightTensor> ts;
  for (int c = 0; c < y.C; ++c) {
    WeightTensor wt;
    wt.name = "pred_c" + std::to_string(c);
    wt.shape = {(int64_t)y.Ny, (int64_t)y.Nx};
    wt.data.resize((size_t)y.Ny * y.Nx);
    for (int iy = 0; iy < y.Ny; ++iy)
      for (int ix = 0; ix < y.Nx; ++ix) wt.data[iy * y.Nx + ix] = y(c, iy, ix);
    ts.push_back(std::move(wt));
  }
  std::map<std::string, std::string> meta = {
      {"format", "cylfno-safetensors/1"},
      {"kind", "prediction"},
      {"t", std::to_string(t)},
      {"nx", std::to_string(y.Nx)},
      {"ny", std::to_string(y.Ny)},
      {"fields", "pred_c0=ux pred_c1=uy pred_c2=p"}};
  std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  saveWeights(path, ts, meta, false);
  std::printf("wrote prediction to %s\n", path.c_str());
}

// VTK path for step k (0-based: k=0 is the input state). nSteps == 1 keeps
// the configured path for the single prediction (byte-identical to the old
// behavior); everything else gets indexed siblings (pred.vtk -> pred_0000.vtk).
static std::string vtkStepPath(const std::string &base, int k, int nSteps) {
  if (nSteps <= 1 && k != 0) return base;
  auto dot = base.find_last_of('.');
  char suffix[16];
  std::snprintf(suffix, sizeof suffix, "_%04d", k);
  if (dot == std::string::npos) return base + suffix;
  return base.substr(0, dot) + suffix + base.substr(dot);
}

static int runPredict(FoamReader &reader, UniformGrid &grid, FNO2d &model,
                      const CellCenters &centers, double t0, const Config &cfg) {
  Snapshot s = reader.readTime(t0);
  std::printf("t=%.2f nCells=%d\n", s.time, s.nCells);
  stats("ux", s.ux);
  stats("uy", s.uy);
  stats("p", s.p);

  Tensor3 data = grid.createData(s, centers);
  std::printf("grid: C=%d Ny=%d Nx=%d\n", data.C, data.Ny, data.Nx);

  // debug, trying to understand datya structure
  std::printf("Debug output: ux=%f uy=%f p=%f at grid (ix=0,iy=0) = pos (%f,%f)\n",
              data(0, 0, 0), data(1, 0, 0), data(2, 0, 0), grid.cellX(0),
              grid.cellY(0));

  // Step 0: the input state itself, always written when VTK output is on --
  // the fixed reference to compare every prediction frame against.
  if (!cfg.prediction_vtk.empty()) {
    const std::string vp0 = vtkStepPath(cfg.prediction_vtk, 0, cfg.nSteps);
    writeVTK(vp0, data, grid, t0);
    std::printf("wrote input state to %s\n", vp0.c_str());
  }

  Tensor3 x = buildModelInput(data, grid, cfg.fno_channelIn);
  double t = t0;
  Tensor3 y;

  // prediction loop
  for (int k = 1; k <= cfg.nSteps; ++k) {
    y = model.forward(x);
    t += cfg.dt;
    std::printf("step %d/%d t=%.2f: fno out C=%d\n", k, cfg.nSteps, t, y.C);

    // mid loop IO
    if (!cfg.prediction_vtk.empty() &&
        (k % cfg.outputInterval == 0 || k == cfg.nSteps)) {
      const std::string vp = vtkStepPath(cfg.prediction_vtk, k, cfg.nSteps);
      writeVTK(vp, y, grid, t);
      std::printf("wrote ParaView file to %s\n", vp.c_str());
    }

    // recursive looping
    if (k < cfg.nSteps) x = feedbackInput(y, grid, cfg.fno_channelIn);
  }
  
  if (!cfg.prediction_file.empty()) writePrediction(cfg.prediction_file, y, t);
  return 0;
}


static void stats(const char *name, const std::vector<double> &v) {
  if (v.empty()) {
    std::printf("%s: empty\n", name);
    return;
  }
  double mn = v[0], mx = v[0], s = 0;
  for (double x : v) {
    mn = std::min(mn, x);
    mx = std::max(mx, x);
    s += x;
  }
  std::printf("%s: n=%zu min=%.5f max=%.5f mean=%.5f\n", name, v.size(), mn, mx,
              s / v.size());
}

int main(int argc, char **argv) {
  const Config cfg = loadConfig("properties.toml");
  const std::string cmd = argc > 1 ? argv[1] : "";

  // Resolve caseDir per subcommand (CLI overrides properties.toml).
  std::string caseDir = cfg.dataDir;
  std::vector<double> probeTimes;
  unsigned seed = 42; // change for a different random cell
  double t = cfg.predictionStartTimestep;

  // Single instantiation shared by all paths (mesh parsed once).
  FoamReader reader(caseDir);
  
  UniformGrid grid(cfg.nx, cfg.ny, cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax,
                   cfg.cylinder_x, cfg.cylinder_y, cfg.cylinder_r);

  FNO2d model = makeModel(cfg);

  CellCenters centers = loadCellCenters(caseDir); // throws if mesh unreadable

  return runPredict(reader, grid, model, centers, t, cfg);
}
