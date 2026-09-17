#pragma once
// FoamReader: parse OpenFOAM ascii volVectorField (U) and volScalarField (p).
// Handles `internalField uniform ...` and `internalField nonuniform List<...>`.
#include <array>
#include <string>
#include <vector>

struct Snapshot {
  double time = 0.0;
  int nCells = 0;
  std::vector<double> ux, uy, uz;  // size nCells
  std::vector<double> p;           // size nCells
};

class FoamReader {
 public:
  explicit FoamReader(std::string caseDir);  // e.g. "data/run/2D_cylinder"

  // Read U and p at one time directory (e.g. t=28 reads "28/U" and "28/p").
  Snapshot readTime(double t);

  // Scan case dir for numeric time folders (0,2,...,40).
  std::vector<double> times();

 private:
  std::string caseDir_;
  std::string timePath(double t);

  // Low-level parsers (ascii only).
  static std::vector<std::array<double, 3>> parseVectorInternalField(
      const std::string& file);
  static std::vector<double> parseScalarInternalField(const std::string& file);
};
