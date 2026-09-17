#pragma once
// UniformGrid: resample unstructured OpenFOAM cells -> uniform Nx*Ny tensor
// for the FFT in the FNO (Li et al. 2021, Sec. 4 assumes uniform grid).
// Layout: Tensor3 { C, Ny, Nx }, row-major, channel-first:
//   ch 0 = ux, ch 1 = uy, ch 2 = p (+ optional x,y coords appended by FNO).
#include <vector>

#include "foam_reader.hpp"

struct Tensor3 {
  int C = 0, Ny = 0, Nx = 0;
  std::vector<double> d;  // size C*Ny*Nx
  double& operator()(int c, int iy, int ix) {
    return d[(c * Ny + iy) * Nx + ix];
  }
  const double& operator()(int c, int iy, int ix) const {
    return d[(c * Ny + iy) * Nx + ix];
  }
};

struct CellCenters {
  std::vector<double> x, y;  // size nCells
};

class UniformGrid {
 public:
  UniformGrid(int nx = 64, int ny = 64, double xmin = 0, double xmax = 10,
              double ymin = 0, double ymax = 10);

  // Nearest-neighbor resample of one snapshot onto the uniform grid.
  // centers: cell-center coords matching snapshot ordering.
  Tensor3 resample(const Snapshot& s, const CellCenters& centers);

  // 1 inside fluid, 0 inside cylinder/solid (for masked loss later).
  // Analytic cylinder: center (3,5), radius 1 — see
  // system/snappyHexMeshDict `geometry { cylinder { ... } }`.
  // TODO: parse the dict instead of hardcoding.
  Tensor3 mask() const;

  // Normalized (x,y) coordinate channels in [0,1], as in Li Sec. 5.3.
  Tensor3 coords() const;

  int nx() const { return nx_; }
  int ny() const { return ny_; }

 private:
  int nx_, ny_;
  double xmin_, xmax_, ymin_, ymax_;
};

// Reads constant/polyMesh/{points,faces,owner,neighbour} and returns the
// (x,y) centroid of every cell by averaging adjacent face centers.
// Throws on missing files or size mismatches.
CellCenters loadCellCenters(const std::string& caseDir);
