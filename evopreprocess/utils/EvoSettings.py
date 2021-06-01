"""
Helper class for default setting of some evolutionary and nature inspired NiaPy algorithms.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: GNU General Public License v3.0

import niapy.algorithms.basic as nia

population_size = 10
all_kwargs = {'max_evals': 1000}
ga_kwargs = {'population_size': population_size}
bat_kwargs = {'population_size': population_size}
pso_kwargs = {'population_size': population_size}
es_kwargs = {'population_size': population_size}
abc_kwargs = {'population_size': population_size}
de_kwargs = {'population_size': population_size}
fa_kwargs = {'population_size': population_size}
gw_kwargs = {'population_size': population_size}
hs_kwargs = {'population_size': population_size}
fw_kwargs = {'population_size': population_size}

kwargs = {nia.GeneticAlgorithm: ga_kwargs,
          nia.BatAlgorithm: bat_kwargs,
          nia.ParticleSwarmAlgorithm: pso_kwargs,
          nia.EvolutionStrategyMpL: es_kwargs,
          nia.ArtificialBeeColonyAlgorithm: abc_kwargs,
          nia.DifferentialEvolution: de_kwargs,
          nia.FireflyAlgorithm: fa_kwargs,
          nia.GreyWolfOptimizer: gw_kwargs,
          nia.HarmonySearch: hs_kwargs,
          nia.BareBonesFireworksAlgorithm: fw_kwargs}


def get_args(optimizer):
    """
    Method returns dictionary with default setting for provided optimization method.

    Parameters
    ----------
    optimizer : Optimization method for which default settings will be returned.

    Returns
    -------
    args : Dictionary of default settings for the provided optimization method.
    """
    args = {**all_kwargs,
            **kwargs.get(optimizer, {})}
    return args
