#include "vtk.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>

void writeVTK(const std::string& path, const Tensor3& y,
              const UniformGrid& grid, double t) {
  if (y.C < 3) throw std::runtime_error("writeVTK: need >= 3 channels");
  if (y.Ny != grid.ny() || y.Nx != grid.nx())
    throw std::runtime_error("writeVTK: tensor/grid size mismatch");
  const double dx = (grid.xmax() - grid.xmin()) / grid.nx();
  const double dy = (grid.ymax() - grid.ymin()) / grid.ny();
  Tensor3 mask = grid.mask();

  std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("writeVTK: cannot open " + path);
  std::fprintf(f, "# vtk DataFile Version 2.0\n");
  std::fprintf(f, "cylfno prediction t=%.2f\n", t);
  std::fprintf(f, "ASCII\n");
  std::fprintf(f, "DATASET STRUCTURED_POINTS\n");
  std::fprintf(f, "DIMENSIONS %d %d 1\n", grid.nx() + 1, grid.ny() + 1);
  std::fprintf(f, "ORIGIN %.6f %.6f 0\n", grid.xmin(), grid.ymin());
  std::fprintf(f, "SPACING %.6f %.6f 1\n", dx, dy);
  std::fprintf(f, "CELL_DATA %d\n", grid.nx() * grid.ny());
  std::fprintf(f, "VECTORS velocity double\n");
  for (int iy = 0; iy < grid.ny(); ++iy)
    for (int ix = 0; ix < grid.nx(); ++ix)
      std::fprintf(f, "%.6f %.6f 0\n", y(0, iy, ix), y(1, iy, ix));
  std::fprintf(f, "FIELD CellFields 2\n");
  std::fprintf(f, "pressure 1 %d double\n", grid.nx() * grid.ny());
  for (int iy = 0; iy < grid.ny(); ++iy)
    for (int ix = 0; ix < grid.nx(); ++ix)
      std::fprintf(f, "%.6f\n", y(2, iy, ix));
  std::fprintf(f, "mask 1 %d double\n", grid.nx() * grid.ny());
  for (int iy = 0; iy < grid.ny(); ++iy)
    for (int ix = 0; ix < grid.nx(); ++ix)
      std::fprintf(f, "%.0f\n", mask(0, iy, ix));
  if (std::fclose(f) != 0) throw std::runtime_error("writeVTK: write failed");
}
