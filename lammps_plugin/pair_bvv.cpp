/* ----------------------------------------------------------------------
   Bond-valence vector (BVV) pair style — see pair_bvv.h.

   Reference implementation: M3L/bv-ff/src/potentials.py (class BVV).
   The force formulas are a literal transcription of BVV._pair_engine:

     c_x   = D_x (|W_x|^2 - W0_x^2)
     v     = c_i W_i - c_j W_j                    (per pair, rhat = (x_j-x_i)/r)
     df_i  = 4 (dV - V/r)(v.rhat) rhat + 4 (V/r) v
     f_j   = -df_i

   Parameter file (whitespace/# tolerant), written by scripts/export_lammps.py:

     cutoff        6.0
     smooth_width  1.0
     form          exp            # exp | power
     species       2
     Pb  1.553448  0.178537      # name W0 D
     Ti  0.278416  0.098188
     pairs         2
     O Pb  2.072416 6.0 0.495635  # name1 name2 r0 C b
     O Ti  1.760536 5.2 0.377942
------------------------------------------------------------------------- */

#include "pair_bvv.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairBVV::PairBVV(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;      // many-body: no isolated pair energy
  restartinfo = 0;
  one_coeff = 1;          // pair_coeff * * only
  manybody_flag = 1;

  cut_global = 0.0;
  smooth_width = 0.0;
  form_exp = 1;
  W0 = Dcoef = nullptr;
  pair_on = nullptr;
  r0_tbl = c_tbl = b_tbl = nullptr;
  Wvec = nullptr;
  cscal = nullptr;
  nmax = 0;

  comm_forward = 3;       // W of owned atoms -> ghosts
  comm_reverse = 3;       // ghost W contributions -> owners
}

/* ---------------------------------------------------------------------- */

PairBVV::~PairBVV()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(W0);
    memory->destroy(Dcoef);
    memory->destroy(pair_on);
    memory->destroy(r0_tbl);
    memory->destroy(c_tbl);
    memory->destroy(b_tbl);
  }
  memory->destroy(Wvec);
  memory->destroy(cscal);
}

/* ---------------------------------------------------------------------- */

void PairBVV::allocate()
{
  allocated = 1;
  const int np1 = atom->ntypes + 1;
  memory->create(setflag, np1, np1, "pair:setflag");
  memory->create(cutsq, np1, np1, "pair:cutsq");
  memory->create(W0, np1, "pair:W0");
  memory->create(Dcoef, np1, "pair:Dcoef");
  memory->create(pair_on, np1, np1, "pair:pair_on");
  memory->create(r0_tbl, np1, np1, "pair:r0_tbl");
  memory->create(c_tbl, np1, np1, "pair:c_tbl");
  memory->create(b_tbl, np1, np1, "pair:b_tbl");
  for (int i = 0; i < np1; i++)
    for (int j = 0; j < np1; j++) {
      setflag[i][j] = 0;
      pair_on[i][j] = 0;
      r0_tbl[i][j] = c_tbl[i][j] = b_tbl[i][j] = 0.0;
    }
  for (int i = 0; i < np1; i++) W0[i] = Dcoef[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void PairBVV::settings(int narg, char ** /*arg*/)
{
  if (narg != 0) error->all(FLERR, "pair_style bvv takes no arguments (all parameters come from the coeff file)");
}

/* ---------------------------------------------------------------------- */

void PairBVV::read_file(const char *filename, int ntypes, char **elem_of_type)
{
  std::ifstream in(filename);
  if (!in) error->all(FLERR, "Cannot open BVV parameter file {}", filename);

  std::map<std::string, std::pair<double, double>> species;              // name -> (W0, D)
  std::map<std::pair<std::string, std::string>, std::array<double, 3>> pairs;

  auto next_tokens = [&in](std::vector<std::string> &tok) -> bool {
    std::string line;
    while (std::getline(in, line)) {
      const auto h = line.find('#');
      if (h != std::string::npos) line.erase(h);
      std::istringstream ss(line);
      tok.clear();
      std::string w;
      while (ss >> w) tok.push_back(w);
      if (!tok.empty()) return true;
    }
    return false;
  };

  std::vector<std::string> tok;
  while (next_tokens(tok)) {
    if (tok[0] == "cutoff" && tok.size() == 2) {
      cut_global = std::stod(tok[1]);
    } else if (tok[0] == "smooth_width" && tok.size() == 2) {
      smooth_width = std::stod(tok[1]);
    } else if (tok[0] == "form" && tok.size() == 2) {
      if (tok[1] == "exp") form_exp = 1;
      else if (tok[1] == "power") form_exp = 0;
      else error->all(FLERR, "BVV form must be 'exp' or 'power', got {}", tok[1]);
    } else if (tok[0] == "species" && tok.size() == 2) {
      const int n = std::stoi(tok[1]);
      for (int k = 0; k < n; k++) {
        if (!next_tokens(tok) || tok.size() != 3)
          error->all(FLERR, "BVV file: expected 'name W0 D' species line");
        species[tok[0]] = {std::stod(tok[1]), std::stod(tok[2])};
      }
    } else if (tok[0] == "pairs" && tok.size() == 2) {
      const int n = std::stoi(tok[1]);
      for (int k = 0; k < n; k++) {
        if (!next_tokens(tok) || tok.size() != 5)
          error->all(FLERR, "BVV file: expected 'name1 name2 r0 C b' pair line");
        pairs[{tok[0], tok[1]}] = {std::stod(tok[2]), std::stod(tok[3]), std::stod(tok[4])};
      }
    } else {
      error->all(FLERR, "BVV file: unrecognized line starting with '{}'", tok[0]);
    }
  }
  if (cut_global <= 0.0) error->all(FLERR, "BVV file must set a positive cutoff");

  for (int t = 1; t <= ntypes; t++) {
    if (strcmp(elem_of_type[t], "NULL") == 0) continue;
    const auto s = species.find(elem_of_type[t]);
    if (s != species.end()) {
      W0[t] = s->second.first;
      Dcoef[t] = s->second.second;
    }
  }
  for (int t1 = 1; t1 <= ntypes; t1++) {
    for (int t2 = 1; t2 <= ntypes; t2++) {
      if (strcmp(elem_of_type[t1], "NULL") == 0 || strcmp(elem_of_type[t2], "NULL") == 0) continue;
      auto p = pairs.find({elem_of_type[t1], elem_of_type[t2]});
      if (p == pairs.end()) p = pairs.find({elem_of_type[t2], elem_of_type[t1]});
      if (p != pairs.end()) {
        pair_on[t1][t2] = 1;
        r0_tbl[t1][t2] = p->second[0];
        c_tbl[t1][t2] = p->second[1];
        b_tbl[t1][t2] = p->second[2];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairBVV::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  const int ntypes = atom->ntypes;
  if (narg != 3 + ntypes)
    error->all(FLERR, "pair_coeff bvv needs: * * file.bvv elem-per-type... ({} types)", ntypes);
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff bvv must be given as 'pair_coeff * *'");

  read_file(arg[2], ntypes, &arg[2]);   // arg[3..] indexed as elem_of_type[1..]

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
}

/* ---------------------------------------------------------------------- */

void PairBVV::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "pair_style bvv requires newton pair on");
  neighbor->add_request(this);
}

/* ---------------------------------------------------------------------- */

double PairBVV::init_one(int /*i*/, int /*j*/)
{
  return cut_global;
}

/* ---------------------------------------------------------------------- */

inline void PairBVV::switch_fn(double r, double &S, double &dS) const
{
  S = 1.0;
  dS = 0.0;
  if (smooth_width <= 0.0) return;
  const double x = (r - (cut_global - smooth_width)) / smooth_width;
  if (x <= 0.0) return;
  if (x >= 1.0) {
    S = 0.0;
    return;
  }
  S = 1.0 + x * x * x * (-10.0 + x * (15.0 - 6.0 * x));
  dS = x * x * (-30.0 + x * (60.0 - 30.0 * x)) / smooth_width;
}

/* V_ij(r) and dV/dr for the pair (itype, jtype), taper included. */

inline void PairBVV::valence(int itype, int jtype, double r, double &V, double &dV) const
{
  const double r0 = r0_tbl[itype][jtype];
  double Vr, dVr;
  if (form_exp) {
    const double b = b_tbl[itype][jtype];
    Vr = std::exp((r0 - r) / b);
    dVr = -Vr / b;
  } else {
    const double C = c_tbl[itype][jtype];
    Vr = std::pow(r0 / r, C);
    dVr = -C * Vr / r;
  }
  double S, dS;
  switch_fn(r, S, dS);
  V = Vr * S;
  dV = dVr * S + Vr * dS;
}

/* ---------------------------------------------------------------------- */

void PairBVV::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (atom->nmax > nmax) {
    memory->destroy(Wvec);
    memory->destroy(cscal);
    nmax = atom->nmax;
    memory->create(Wvec, nmax, 3, "pair:Wvec");
    memory->create(cscal, nmax, "pair:cscal");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const double cutsq_g = cut_global * cut_global;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // ── pass 1: vector density W_i = sum_j V_ij rhat_ij (rhat from i to j) ──
  for (int i = 0; i < nall; i++) Wvec[i][0] = Wvec[i][1] = Wvec[i][2] = 0.0;

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int itype = type[i];
    const double xtmp = x[i][0], ytmp = x[i][1], ztmp = x[i][2];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      const int jtype = type[j];
      if (!pair_on[itype][jtype]) continue;
      const double delx = xtmp - x[j][0];
      const double dely = ytmp - x[j][1];
      const double delz = ztmp - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq_g) continue;
      const double r = std::sqrt(rsq);
      double V, dV;
      valence(itype, jtype, r, V, dV);
      const double vr = V / r;
      // rhat_ij = (x_j - x_i)/r = (-delx, -dely, -delz)/r
      Wvec[i][0] -= vr * delx;
      Wvec[i][1] -= vr * dely;
      Wvec[i][2] -= vr * delz;
      Wvec[j][0] += vr * delx;
      Wvec[j][1] += vr * dely;
      Wvec[j][2] += vr * delz;
    }
  }

  comm->reverse_comm(this);   // sum ghost W into owners

  // per-atom energy (embedding-like) + c = D (|W|^2 - W0^2) for owned atoms
  for (int i = 0; i < nlocal; i++) {
    const int t = type[i];
    const double W2 =
        Wvec[i][0] * Wvec[i][0] + Wvec[i][1] * Wvec[i][1] + Wvec[i][2] * Wvec[i][2];
    const double diff = W2 - W0[t] * W0[t];
    cscal[i] = Dcoef[t] * diff;
    if (eflag) {
      const double phi = Dcoef[t] * diff * diff;
      if (eflag_global) eng_vdwl += phi;
      if (eflag_atom) eatom[i] += phi;
    }
  }

  comm->forward_comm(this);   // W of owners -> ghosts

  for (int i = nlocal; i < nall; i++) {
    const int t = type[i];
    const double W2 =
        Wvec[i][0] * Wvec[i][0] + Wvec[i][1] * Wvec[i][1] + Wvec[i][2] * Wvec[i][2];
    cscal[i] = Dcoef[t] * (W2 - W0[t] * W0[t]);
  }

  // ── pass 2: pair forces (literal transcription of BVV._pair_engine) ──
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int itype = type[i];
    const double xtmp = x[i][0], ytmp = x[i][1], ztmp = x[i][2];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      const int jtype = type[j];
      if (!pair_on[itype][jtype]) continue;
      const double delx = xtmp - x[j][0];
      const double dely = ytmp - x[j][1];
      const double delz = ztmp - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq >= cutsq_g) continue;
      const double r = std::sqrt(rsq);
      double V, dV;
      valence(itype, jtype, r, V, dV);

      const double rhx = -delx / r, rhy = -dely / r, rhz = -delz / r;
      const double vx = cscal[i] * Wvec[i][0] - cscal[j] * Wvec[j][0];
      const double vy = cscal[i] * Wvec[i][1] - cscal[j] * Wvec[j][1];
      const double vz = cscal[i] * Wvec[i][2] - cscal[j] * Wvec[j][2];
      const double vdotr = vx * rhx + vy * rhy + vz * rhz;
      const double radial = 4.0 * (dV - V / r) * vdotr;
      const double lateral = 4.0 * V / r;

      const double fx = radial * rhx + lateral * vx;   // force on i
      const double fy = radial * rhy + lateral * vy;
      const double fz = radial * rhz + lateral * vz;

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      f[j][0] -= fx;
      f[j][1] -= fy;
      f[j][2] -= fz;

      if (evflag)
        ev_tally_xyz(i, j, nlocal, force->newton_pair, 0.0, 0.0, fx, fy, fz, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ── W communication ─────────────────────────────────────────────────── */

int PairBVV::pack_forward_comm(int n, int *list_, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int m = 0;
  for (int k = 0; k < n; k++) {
    const int j = list_[k];
    buf[m++] = Wvec[j][0];
    buf[m++] = Wvec[j][1];
    buf[m++] = Wvec[j][2];
  }
  return m;
}

void PairBVV::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  for (int i = first; i < first + n; i++) {
    Wvec[i][0] = buf[m++];
    Wvec[i][1] = buf[m++];
    Wvec[i][2] = buf[m++];
  }
}

int PairBVV::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  for (int i = first; i < first + n; i++) {
    buf[m++] = Wvec[i][0];
    buf[m++] = Wvec[i][1];
    buf[m++] = Wvec[i][2];
  }
  return m;
}

void PairBVV::unpack_reverse_comm(int n, int *list_, double *buf)
{
  int m = 0;
  for (int k = 0; k < n; k++) {
    const int j = list_[k];
    Wvec[j][0] += buf[m++];
    Wvec[j][1] += buf[m++];
    Wvec[j][2] += buf[m++];
  }
}
