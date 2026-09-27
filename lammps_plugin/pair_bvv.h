/* -*- c++ -*- ----------------------------------------------------------
   Bond-valence vector (BVV) pair style for the M3L BVFF ferroelectric
   force field:

       E = sum_i D_i (|W_i|^2 - W0_i^2)^2,   W_i = sum_j V_ij(r) rhat_ij

   with V_ij either the Brown-Altermatt exponential exp((r0-r)/b) or the
   Brown power law (r0/r)^C, tapered by the same C2 cutoff switch as the
   Python reference implementation (bvff/core/potentials.py).

   EAM-like two-pass structure: accumulate the per-atom vector density W,
   reverse-communicate ghost contributions, then a pair force pass using
   both W_i and W_j (forward-communicated). Matches the Python
   BVV._pair_engine formulas exactly.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(bvv,PairBVV);
// clang-format on
#else

#ifndef LMP_PAIR_BVV_H
#define LMP_PAIR_BVV_H

#include "pair.h"

namespace LAMMPS_NS {

class PairBVV : public Pair {
 public:
  PairBVV(class LAMMPS *);
  ~PairBVV() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  double cut_global;      // cutoff (A)
  double smooth_width;    // C2 taper width (A); 0 = plain truncation
  int form_exp;           // 1 = "exp" (Brown-Altermatt), 0 = "power"

  // per-type species parameters
  double *W0, *Dcoef;     // [ntypes+1]
  // per-type-pair bond-valence shape parameters (0/1 flag + r0, C, b)
  int **pair_on;
  double **r0_tbl, **c_tbl, **b_tbl;

  // per-atom vector density W and scalar c = D (|W|^2 - W0^2)
  double **Wvec;          // [nmax][3]
  double *cscal;          // [nmax]
  int nmax;

  virtual void allocate();
  void read_file(const char *, int, char **);
  inline void valence(int, int, double, double &, double &) const;
  inline void switch_fn(double, double &, double &) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
