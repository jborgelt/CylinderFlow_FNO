#pragma once
// vtk: minimal legacy-ASCII VTK writer for ParaView (5.10 reads it natively).
// Writes the uniform prediction grid as STRUCTURED_POINTS with CELL_DATA,
// which matches our cell-centered Tensor3 exactly (DIMENSIONS nx+1 ny+1 1).
// Fields: VECTORS velocity (ux uy 0, active vectors for glyphs/streamlines)
// plus a FIELD section with pressure + mask (1 fluid, 0 cylinder, for
// Threshold/Clip). NOTE: repeated plain SCALARS blocks get dropped by VTK's
// legacy reader (only one survives), hence FIELD.
// Zero dependencies. Visualization only -- never read back by the model.
#include <string>

#include "grid.hpp"

// y: model output [C,Ny,Nx], first 3 channels are ux,uy,p (extra channels
// are ignored). grid supplies domain + cylinder mask. t: snapshot time.
void writeVTK(const std::string& path, const Tensor3& y,
              const UniformGrid& grid, double t);
