#include "foam_reader.hpp"

#include <array>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

FoamReader::FoamReader(std::string caseDir) : caseDir_(std::move(caseDir)) {}

std::string FoamReader::timePath(double t) {
  // OpenFOAM here uses integer-like names ("0","2",...,"28").
  // Format without trailing .0 so "28" stays "28".
  std::ostringstream os;
  os << t;
  return caseDir_ + "/" + os.str();
}

std::vector<double> FoamReader::times() {
  std::vector<double> out;
  for (const auto& e : fs::directory_iterator(caseDir_)) {
    if (!e.is_directory()) continue;
    const std::string name = e.path().filename().string();
    try {
      size_t pos = 0;
      double t = std::stod(name, &pos);
      if (pos == name.size()) out.push_back(t);
    } catch (...) {
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

static std::string readWhole(const std::string& file) {
  std::ifstream f(file);
  if (!f) throw std::runtime_error("cannot open " + file);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Find "internalField" and return everything after it.
static std::string afterInternalField(const std::string& text) {
  auto pos = text.find("internalField");
  if (pos == std::string::npos) throw std::runtime_error("no internalField");
  return text.substr(pos + std::string("internalField").size());
}

std::vector<std::array<double, 3>>
FoamReader::parseVectorInternalField(const std::string& file) {
  std::string s = afterInternalField(readWhole(file));
  std::istringstream is(s);
  std::string kind;
  is >> kind;
  if (kind == "uniform") {
    // uniform (1 0 0)
    char c;
    double x, y, z;
    is >> c >> x >> y >> z >> c;  // '(' ... ')'
    // nCells unknown here -> caller for t=0 expands later; return single entry
    return {{{x, y, z}}};
  }
  // nonuniform List<vector> N ( ... )
  std::string list;
  long n = 0;
  is >> list >> n;
  std::vector<std::array<double, 3>> out;
  out.reserve(n > 0 ? n : 0);
  char c;
  is >> c;  // '('
  for (long i = 0; i < n; ++i) {
    double x, y, z;
    is >> c >> x >> y >> z >> c;  // '(' x y z ')'
    out.push_back({x, y, z});
  }
  return out;
}

std::vector<double> FoamReader::parseScalarInternalField(
    const std::string& file) {
  std::string s = afterInternalField(readWhole(file));
  std::istringstream is(s);
  std::string kind;
  is >> kind;
  if (kind == "uniform") {
    double v;
    is >> v;
    return {v};
  }
  std::string list;
  long n = 0;
  is >> list >> n;
  std::vector<double> out;
  out.reserve(n > 0 ? n : 0);
  char c;
  is >> c;  // '('
  for (long i = 0; i < n; ++i) {
    double v;
    is >> v;
    out.push_back(v);
  }
  return out;
}

Snapshot FoamReader::readTime(double t) {
  const std::string dir = timePath(t);
  auto vecs = parseVectorInternalField(dir + "/U");
  auto scal = parseScalarInternalField(dir + "/p");

  Snapshot s;
  s.time = t;
  // Uniform initial condition (t=0): single value; keep nCells=1.
  // Real snapshots: sizes must match.
  if (vecs.size() == 1 && scal.size() == 1) {
    s.nCells = 1;
  } else {
    if (vecs.size() != scal.size())
      throw std::runtime_error("U/p size mismatch at t=" + std::to_string(t));
    s.nCells = static_cast<int>(vecs.size());
  }
  s.ux.reserve(vecs.size());
  s.uy.reserve(vecs.size());
  s.uz.reserve(vecs.size());
  for (auto& v : vecs) {
    s.ux.push_back(v[0]);
    s.uy.push_back(v[1]);
    s.uz.push_back(v[2]);
  }
  s.p = std::move(scal);
  // Broadcast uniform IC to a single entry (grid step expands if needed).
  return s;
}
