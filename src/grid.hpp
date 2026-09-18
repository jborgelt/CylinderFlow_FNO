#pragma once
// UniformGrid: create uniform Nx*Ny tensor data from unstructured OpenFOAM cells
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
              double ymin = 0, double ymax = 10, double cylX = 3,
              double cylY = 5, double cylR = 1);

  // Create uniform-grid data from one snapshot: bins unstructured cells
  // by center into grid cells (nearest-cell, averaging collisions).
  // centers: cell-center coords matching snapshot ordering.
  Tensor3 createData(const Snapshot& s, const CellCenters& centers);

  // 1 inside fluid, 0 inside cylinder/solid (for masked loss later).
  // Cylinder comes from the ctor (defaults: snappyHexMeshDict
  // `geometry { cylinder { ... } }`: center (3,5), radius 1).
  Tensor3 mask() const;

  // Normalized (x,y) coordinate channels in [0,1], as in Li Sec. 5.3.
  Tensor3 coords() const;

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  double xmin() const { return xmin_; }
  double xmax() const { return xmax_; }
  double ymin() const { return ymin_; }
  double ymax() const { return ymax_; }
  // Physical coords of a grid cell center.
  double cellX(int ix) const {
    return xmin_ + (ix + 0.5) * (xmax_ - xmin_) / nx_;
  }
  double cellY(int iy) const {
    return ymin_ + (iy + 0.5) * (ymax_ - ymin_) / ny_;
  }

 private:
  int nx_, ny_;
  double xmin_, xmax_, ymin_, ymax_;
  double cylX_, cylY_, cylR_;
};

// Reads constant/polyMesh/{points,faces,owner,neighbour} and returns the
// (x,y) centroid of every cell by averaging adjacent face centers.
// Throws on missing files or size mismatches.
CellCenters loadCellCenters(const std::string& caseDir);
