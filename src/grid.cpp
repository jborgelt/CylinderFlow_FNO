#include "grid.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Minimal OpenFOAM mesh-file helpers (ascii only).
// A mesh file = FoamFile header, then a lone count line, then a (...) list.
// The header never contains a line with only digits, so the first such line
// is the count. Faces glue the count to the paren ("4(1 2 3 4)"), hence the
// tokenizer below that separates '(' and ')'.
// ---------------------------------------------------------------------------

static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r");
  return s.substr(a, b - a + 1);
}

static bool isDigits(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s)
    if (c < '0' || c > '9') return false;
  return true;
}

// Split file into: count (first lone-number line) + body (everything after).
static std::pair<long, std::string> splitCountBody(const std::string& file) {
  std::ifstream f(file);
  if (!f) throw std::runtime_error("cannot open " + file);
  std::string line, body;
  long count = -1;
  bool found = false;
  std::ostringstream rest;
  while (std::getline(f, line)) {
    if (!found && isDigits(trim(line))) {
      count = std::stol(trim(line));
      found = true;
      continue;
    }
    if (found) rest << line << "\n";
  }
  if (!found) throw std::runtime_error("no count line in " + file);
  body = rest.str();
  return {count, body};
}

static std::vector<std::string> tokenize(const std::string& body) {
  std::string spaced;
  spaced.reserve(body.size() * 2);
  for (char c : body) {
    if (c == '(' || c == ')') {
      spaced += ' ';
      spaced += c;
      spaced += ' ';
    } else {
      spaced += c;
    }
  }
  std::istringstream is(spaced);
  std::vector<std::string> toks;
  std::string t;
  while (is >> t) toks.push_back(t);
  return toks;
}

struct Point3 {
  double x, y, z;
};

static std::vector<Point3> readPoints(const std::string& file) {
  auto [count, body] = splitCountBody(file);
  auto toks = tokenize(body);
  std::vector<Point3> pts;
  pts.reserve(count);
  size_t i = 0;
  if (toks[i++] != "(") throw std::runtime_error("points: expected '('");
  for (long n = 0; n < count; ++n) {
    if (toks[i++] != "(") throw std::runtime_error("points: expected '('");
    double x = std::stod(toks[i++]);
    double y = std::stod(toks[i++]);
    double z = std::stod(toks[i++]);
    if (toks[i++] != ")") throw std::runtime_error("points: expected ')'");
    pts.push_back({x, y, z});
  }
  return pts;
}

static std::vector<std::vector<int>> readFaces(const std::string& file) {
  auto [count, body] = splitCountBody(file);
  auto toks = tokenize(body);
  std::vector<std::vector<int>> faces;
  faces.reserve(count);
  size_t i = 0;
  if (toks[i++] != "(") throw std::runtime_error("faces: expected '('");
  for (long n = 0; n < count; ++n) {
    int k = std::stoi(toks[i++]);
    if (toks[i++] != "(") throw std::runtime_error("faces: expected '('");
    std::vector<int> fp;
    fp.reserve(k);
    for (int j = 0; j < k; ++j) fp.push_back(std::stoi(toks[i++]));
    if (toks[i++] != ")") throw std::runtime_error("faces: expected ')'");
    faces.push_back(std::move(fp));
  }
  return faces;
}

static std::vector<int> readLabels(const std::string& file) {
  auto [count, body] = splitCountBody(file);
  auto toks = tokenize(body);
  std::vector<int> labels;
  labels.reserve(count);
  size_t i = 0;
  if (toks[i++] != "(") throw std::runtime_error("labels: expected '('");
  for (long n = 0; n < count; ++n) labels.push_back(std::stoi(toks[i++]));
  return labels;
}

// ---------------------------------------------------------------------------

UniformGrid::UniformGrid(int nx, int ny, double xmin, double xmax, double ymin,
                         double ymax, double cylX, double cylY, double cylR)
    : nx_(nx),
      ny_(ny),
      xmin_(xmin),
      xmax_(xmax),
      ymin_(ymin),
      ymax_(ymax),
      cylX_(cylX),
      cylY_(cylY),
      cylR_(cylR) {}

Tensor3 UniformGrid::resample(const Snapshot& s, const CellCenters& centers) {
  if ((size_t)s.nCells != centers.x.size() || s.nCells == 0)
    throw std::runtime_error(
        "resample: need cell centers matching the snapshot "
        "(call loadCellCenters first)");
  Tensor3 out;
  out.C = 3;
  out.Ny = ny_;
  out.Nx = nx_;
  out.d.assign((size_t)3 * ny_ * nx_, 0.0);
  std::vector<int> cnt((size_t)ny_ * nx_, 0);

  const double dx = (xmax_ - xmin_) / nx_;
  const double dy = (ymax_ - ymin_) / ny_;

  for (int i = 0; i < s.nCells; ++i) {
    int ix = (int)((centers.x[i] - xmin_) / dx);
    int iy = (int)((centers.y[i] - ymin_) / dy);
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (ix >= nx_) ix = nx_ - 1;
    if (iy >= ny_) iy = ny_ - 1;
    out(0, iy, ix) += s.ux[i];
    out(1, iy, ix) += s.uy[i];
    out(2, iy, ix) += s.p[i];
    cnt[(size_t)iy * nx_ + ix]++;
  }
  for (int iy = 0; iy < ny_; ++iy)
    for (int ix = 0; ix < nx_; ++ix)
      if (cnt[(size_t)iy * nx_ + ix] > 1) {
        const int n = cnt[(size_t)iy * nx_ + ix];
        out(0, iy, ix) /= n;
        out(1, iy, ix) /= n;
        out(2, iy, ix) /= n;
      }
  return out;
}

Tensor3 UniformGrid::mask() const {
  Tensor3 m;
  m.C = 1;
  m.Ny = ny_;
  m.Nx = nx_;
  m.d.assign((size_t)ny_ * nx_, 1.0);
  const double dx = (xmax_ - xmin_) / nx_;
  const double dy = (ymax_ - ymin_) / ny_;
  for (int iy = 0; iy < ny_; ++iy)
    for (int ix = 0; ix < nx_; ++ix) {
      const double cx = xmin_ + (ix + 0.5) * dx;
      const double cy = ymin_ + (iy + 0.5) * dy;
      const double dist = std::sqrt((cx - cylX_) * (cx - cylX_) +
                                    (cy - cylY_) * (cy - cylY_));
      if (dist < cylR_) m(0, iy, ix) = 0.0;
    }
  return m;
}

Tensor3 UniformGrid::coords() const {
  Tensor3 c;
  c.C = 2;
  c.Ny = ny_;
  c.Nx = nx_;
  c.d.resize((size_t)2 * ny_ * nx_);
  for (int iy = 0; iy < ny_; ++iy)
    for (int ix = 0; ix < nx_; ++ix) {
      c(0, iy, ix) = (nx_ == 1) ? 0 : (double)ix / (nx_ - 1);  // x in [0,1]
      c(1, iy, ix) = (ny_ == 1) ? 0 : (double)iy / (ny_ - 1);  // y in [0,1]
    }
  return c;
}

CellCenters loadCellCenters(const std::string& caseDir) {
  const std::string pm = caseDir + "/constant/polyMesh/";
  auto points = readPoints(pm + "points");
  auto faces = readFaces(pm + "faces");
  auto owner = readLabels(pm + "owner");
  auto neighbour = readLabels(pm + "neighbour");  // internal faces only
  if (faces.size() != owner.size())
    throw std::runtime_error("mesh: faces/owner size mismatch");

  int nCells = 0;
  for (int o : owner) nCells = std::max(nCells, o + 1);
  for (int nb : neighbour) nCells = std::max(nCells, nb + 1);

  std::vector<double> sx(nCells, 0.0), sy(nCells, 0.0);
  std::vector<int> cnt(nCells, 0);
  for (size_t fi = 0; fi < faces.size(); ++fi) {
    double fx = 0, fy = 0;
    for (int pi : faces[fi]) {
      fx += points[pi].x;
      fy += points[pi].y;
    }
    fx /= faces[fi].size();
    fy /= faces[fi].size();
    int o = owner[fi];
    sx[o] += fx;
    sy[o] += fy;
    cnt[o]++;
    if (fi < neighbour.size()) {  // internal face: shared with neighbour cell
      int nb = neighbour[fi];
      sx[nb] += fx;
      sy[nb] += fy;
      cnt[nb]++;
    }
  }
  CellCenters cc;
  cc.x.resize(nCells);
  cc.y.resize(nCells);
  for (int c = 0; c < nCells; ++c) {
    cc.x[c] = sx[c] / cnt[c];
    cc.y[c] = sy[c] / cnt[c];
  }
  return cc;
}
