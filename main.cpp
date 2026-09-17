// Driver:
//   ./cylfno [caseDir] [time]  -> single snapshot: stats + FNO forward + mse
//   ./cylfno inspect [caseDir] -> load ALL snapshots, build next-step pairs,
//                                 debug-print the data structures
//   ./cylfno probe [caseDir] [seed SEED] t1 t2 ... -> one random grid cell
//                                 printed across the given timesteps
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "fno.hpp"
#include "foam_reader.hpp"
#include "grid.hpp"
#include "train.hpp"

static void stats(const char* name, const std::vector<double>& v) {
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

static int runInspect(const std::string& caseDir) {
  FoamReader reader(caseDir);
  auto times = reader.times();
  std::printf("case: %s  snapshots:", caseDir.c_str());
  for (double t : times) std::printf(" %.0f", t);
  std::printf("\n");

  CellCenters centers = loadCellCenters(caseDir);
  std::printf("cell centers: n=%zu x∈[%.2f,%.2f] y∈[%.2f,%.2f]\n",
              centers.x.size(), *std::min_element(centers.x.begin(), centers.x.end()),
              *std::max_element(centers.x.begin(), centers.x.end()),
              *std::min_element(centers.y.begin(), centers.y.end()),
              *std::max_element(centers.y.begin(), centers.y.end()));

  UniformGrid grid(64, 64);
  Tensor3 mask = grid.mask();
  double fluid = 0;
  for (double v : mask.d) fluid += v;
  std::printf("mask: %.0f/%.0f cells fluid (rest = cylinder)\n", fluid,
              (double)mask.d.size());

  auto data = buildDataset(reader, grid, centers, times);
  inspectDataset(data);
  return 0;
}

static int runProbe(const std::string& caseDir,
                    const std::vector<double>& times, unsigned seed) {
  FoamReader reader(caseDir);
  CellCenters centers = loadCellCenters(caseDir);
  UniformGrid grid(64, 64);
  Tensor3 mask = grid.mask();

  // One random fluid grid cell, fixed for all timesteps.
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> distX(0, grid.nx() - 1);
  std::uniform_int_distribution<int> distY(0, grid.ny() - 1);
  int px = 0, py = 0;
  for (int tries = 0; tries < 10000; ++tries) {  // redraw if inside cylinder
    px = distX(gen);
    py = distY(gen);
    if (mask(0, py, px) > 0.5) break;
  }
  const double gx = (px + 0.5) * 10.0 / grid.nx();  // domain is 10x10
  const double gy = (py + 0.5) * 10.0 / grid.ny();
  std::printf("probe cell: (ix=%d, iy=%d) at (x=%.4f, y=%.4f), seed=%u\n", px,
              py, gx, gy, seed);

  for (double t : times) {
    Snapshot s = reader.readTime(t);
    if ((size_t)s.nCells != centers.x.size()) {
      std::printf("t=%5.1f: uniform IC (no per-cell data), skipped\n", t);
      continue;
    }
    Tensor3 g = grid.resample(s, centers);
    // Nearest raw cell to the probe point, for comparison.
    int best = 0;
    double bestD = 1e300;
    for (int i = 0; i < s.nCells; ++i) {
      double dx = centers.x[i] - gx, dy = centers.y[i] - gy;
      double d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    std::printf(
        "t=%5.1f: grid ux=%+.5f uy=%+.5f p=%+.5f | raw cell %d "
        "(x=%.4f,y=%.4f) ux=%+.5f uy=%+.5f p=%+.5f\n",
        t, g(0, py, px), g(1, py, px), g(2, py, px), best, centers.x[best],
        centers.y[best], s.ux[best], s.uy[best], s.p[best]);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "inspect")
    return runInspect(argc > 2 ? argv[2] : "data/run/2D_cylinder");

  if (argc > 1 && std::string(argv[1]) == "probe") {
    // PLAY WITH ME: edit this list, or pass times on the command line:
    //   ./build/cylfno probe data/run/2D_cylinder 28 30 32 40
    std::vector<double> probeTimes = {28, 30, 32, 40};
    unsigned seed = 42;  // change for a different random cell
    std::string caseDir = "data/run/2D_cylinder";
    // Optional CLI override: probe [caseDir] [seed] t1 t2 ...
    int i = 2;
    if (argc > i && std::string(argv[i]).find("run") != std::string::npos)
      caseDir = argv[i++];
    if (argc > i && std::string(argv[i]) == "seed") {
      seed = (unsigned)std::stoul(argv[++i]);
      ++i;
    }
    if (argc > i) {
      probeTimes.clear();
      for (; i < argc; ++i) probeTimes.push_back(std::stod(argv[i]));
    }
    return runProbe(caseDir, probeTimes, seed);
  }

  const std::string caseDir = argc > 1 ? argv[1] : "data/run/2D_cylinder";
  const double t = argc > 2 ? std::stod(argv[2]) : 28.0;

  FoamReader reader(caseDir);
  Snapshot s = reader.readTime(t);
  std::printf("t=%.2f nCells=%d\n", s.time, s.nCells);
  stats("ux", s.ux);
  stats("uy", s.uy);
  stats("p", s.p);

  UniformGrid grid(64, 64);
  CellCenters centers = loadCellCenters(caseDir);  // throws if mesh unreadable
  Tensor3 g = grid.resample(s, centers);
  std::printf("grid: C=%d Ny=%d Nx=%d\n", g.C, g.Ny, g.Nx);

  // Append (x,y) coords -> [5,Ny,Nx] model input, like Li Sec. 5.3.
  Tensor3 c = grid.coords();
  Tensor3 x;
  x.C = 5;
  x.Ny = g.Ny;
  x.Nx = g.Nx;
  x.d.resize((size_t)5 * g.Ny * g.Nx);
  for (int iy = 0; iy < g.Ny; ++iy)
    for (int ix = 0; ix < g.Nx; ++ix) {
      x(0, iy, ix) = g(0, iy, ix);
      x(1, iy, ix) = g(1, iy, ix);
      x(2, iy, ix) = g(2, iy, ix);
      x(3, iy, ix) = c(0, iy, ix);
      x(4, iy, ix) = c(1, iy, ix);
    }

  FNO2d model(5, 32, 3, 12, 4);
  Tensor3 y = model.forward(x);
  std::printf("fno out: C=%d Ny=%d Nx=%d\n", y.C, y.Ny, y.Nx);

  Sample sm;
  sm.tIn = sm.tOut = t;
  sm.x = x;
  sm.y = g;  // scaffold: next-step target = same snapshot
  Trainer tr(model);
  std::printf("mse(self) = %.6f\n", tr.trainStep(sm));
  return 0;
}
