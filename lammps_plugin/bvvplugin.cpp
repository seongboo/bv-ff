/* Plugin loader for the BVV pair style: `plugin load ./bvvplugin.so`
   registers `pair_style bvv` in a PLUGIN-enabled LAMMPS binary. */

#include "lammpsplugin.h"
#include "version.h"

#include "pair_bvv.h"

using namespace LAMMPS_NS;

static Pair *pair_bvv_creator(LAMMPS *lmp)
{
  return new PairBVV(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;
  lammpsplugin_t plugin;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "bvv";
  plugin.info = "bond-valence vector (BVV) pair style for the M3L BVFF v1.0";
  plugin.author = "M3L bv-ff (seongboo)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &pair_bvv_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
