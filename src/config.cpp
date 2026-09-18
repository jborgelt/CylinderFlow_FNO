#include "config.hpp"

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>

namespace {
std::string cfgTrim(const std::string &s) {
  size_t a = s.find_first_not_of(" \t\r");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r");
  return s.substr(a, b - a + 1);
}
}  // namespace

// Minimal TOML subset: `key = value` lines, `#` comments, int/double
// numbers and "quoted" strings.
Config loadConfig(const std::string &path) {
  Config c;
  std::ifstream f(path);
  if (!f) {
    std::printf("note: %s not found, using built-in defaults\n", path.c_str());
    return c;
  }
  auto setInt = [&](const std::string &k, const std::string &v) -> bool {
    try {
      int n = std::stoi(v);
      if (k == "fno_width") c.fno_width = n;
      else if (k == "fno_layers") c.fno_layers = n;
      else if (k == "fno_modes") c.fno_modes = n;
      else if (k == "fno_channelIn") c.fno_channelIn = n;
      else if (k == "fno_channelOut") c.fno_channelOut = n;
      else if (k == "q_hidden") c.q_hidden = n;
      else if (k == "p_layers") c.p_layers = n;
      else if (k == "q_layers") c.q_layers = n;
      else if (k == "nx") c.nx = n;
      else if (k == "ny") c.ny = n;
      else if (k == "nSteps") c.nSteps = n;
      else if (k == "outputInterval") c.outputInterval = n;
      else return false;
      return true;
    } catch (...) {
      return false;
    }
  };
  std::string line;
  std::set<std::string> seen;
  while (std::getline(f, line)) {
    line = cfgTrim(line.substr(0, line.find('#')));
    if (line.empty() || line[0] == '[') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string k = cfgTrim(line.substr(0, eq));
    std::string v = cfgTrim(line.substr(eq + 1));
    seen.insert(k);
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {      if (k == "dataDir") c.dataDir = v.substr(1, v.size() - 2);
      else if (k == "weight_file") c.weight_file = v.substr(1, v.size() - 2);
      else if (k == "weight_precision")
        c.weight_precision = v.substr(1, v.size() - 2);
      else if (k == "prediction_file")
        c.prediction_file = v.substr(1, v.size() - 2);
      else if (k == "prediction_vtk")
        c.prediction_vtk = v.substr(1, v.size() - 2);
      continue;
    }
    if (setInt(k, v)) continue;
    try {
      double d = std::stod(v);
      if (k == "startProbeTime") c.startProbeTime = d;
      else if (k == "endProbeTime") c.endProbeTime = d;
      else if (k == "startProbeInterval") c.startProbeInterval = d;
      else if (k == "predictionStartTimestep") c.predictionStartTimestep = d;
      else if (k == "dt") c.dt = d;
      else if (k == "xmin") c.xmin = d;
      else if (k == "xmax") c.xmax = d;
      else if (k == "ymin") c.ymin = d;
      else if (k == "ymax") c.ymax = d;
      else if (k == "cylinder_x") c.cylinder_x = d;
      else if (k == "cylinder_y") c.cylinder_y = d;
      else if (k == "cylinder_r") c.cylinder_r = d;
    } catch (...) {
    }
  }
  // Warn about every expected key that is missing, so the user knows which
  // properties they may have to specify for their case. The program still
  // runs with the built-in default (deliberately a warning, not an assert,
  // so a partial file stays usable for quick experiments).
  const Config dflt;
  auto check = [&](const char *key, const auto &dfltVal) {
    if (!seen.count(key)) {
      std::ostringstream os;
      os << dfltVal;
      if (os.str().empty()) return;  // opt-in keys (weight_file): stay silent
      std::printf("warning: '%s' missing in %s, using default %s\n", key,
                  path.c_str(), os.str().c_str());
    }
  };
  check("fno_width", dflt.fno_width);
  check("fno_layers", dflt.fno_layers);
  check("fno_modes", dflt.fno_modes);
  check("fno_channelIn", dflt.fno_channelIn);
  check("fno_channelOut", dflt.fno_channelOut);
  check("q_hidden", dflt.q_hidden);
  check("p_layers", dflt.p_layers);
  check("q_layers", dflt.q_layers);
  if (c.p_layers < 1 || c.q_layers < 1)
    throw std::runtime_error("p_layers and q_layers must be >= 1");
  check("dataDir", dflt.dataDir);
  check("startProbeTime", dflt.startProbeTime);
  check("endProbeTime", dflt.endProbeTime);
  check("startProbeInterval", dflt.startProbeInterval);
  check("nx", dflt.nx);
  check("ny", dflt.ny);
  check("xmin", dflt.xmin);
  check("xmax", dflt.xmax);
  check("ymin", dflt.ymin);
  check("ymax", dflt.ymax);
  check("cylinder_x", dflt.cylinder_x);
  check("cylinder_y", dflt.cylinder_y);
  check("cylinder_r", dflt.cylinder_r);
  check("weight_file", dflt.weight_file);
  check("weight_precision", dflt.weight_precision);
  if (c.weight_precision != "float64" && c.weight_precision != "float32")
    throw std::runtime_error("weight_precision must be float64 or float32");
  check("predictionStartTimestep", dflt.predictionStartTimestep);
  check("prediction_file", dflt.prediction_file);
  check("prediction_vtk", dflt.prediction_vtk);
  check("dt", dflt.dt);
  check("nSteps", dflt.nSteps);
  check("outputInterval", dflt.outputInterval);
  if (c.dt <= 0) throw std::runtime_error("dt must be > 0");
  if (c.nSteps < 1) throw std::runtime_error("nSteps must be >= 1");
  if (c.outputInterval < 1)
    throw std::runtime_error("outputInterval must be >= 1");
  return c;
}

std::vector<double> probeRange(const Config &c) {
  std::vector<double> out;
  for (double t = c.startProbeTime; t <= c.endProbeTime + 1e-9;
       t += c.startProbeInterval)
    out.push_back(t);
  return out;
}
