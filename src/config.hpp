#pragma once
// Shared run configuration, loaded from properties.toml.
// main.cpp, grid and fno all read their tunables from here instead of
// hardcoding them. Unknown keys are ignored; missing file/keys fall back
// to the defaults below.
#include <string>
#include <vector>

struct Config {
  // FNO model (Li et al. 2021): lift width, spectral layers, kept modes,
  // input/output channels, Q-projection hidden size.
  int fno_width = 32, fno_layers = 4, fno_modes = 12;
  int fno_channelIn = 5, fno_channelOut = 3;
  int q_hidden = 128;
  // Data: case dir + default probe window.
  std::string dataDir = "data/run/2D_cylinder";
  double startProbeTime = 28, endProbeTime = 40, startProbeInterval = 2;
  // Uniform grid for the FFT + analytic cylinder (snappyHexMesh cutout).
  int nx = 64, ny = 64;
  double xmin = 0, xmax = 10, ymin = 0, ymax = 10;
  double cylinder_x = 3, cylinder_y = 5, cylinder_r = 1;};

Config loadConfig(const std::string &path);

// Timesteps startProbeTime..endProbeTime in startProbeInterval steps.
std::vector<double> probeRange(const Config &c);
