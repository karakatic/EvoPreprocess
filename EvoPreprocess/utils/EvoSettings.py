"""
Helper class for default setting of some evolutionary and nature inspired NiaPy algorithms.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: GNU General Public License v3.0


import NiaPy.algorithms.basic as nia

NP = 10
all_kwargs = {'nFES': 1000}
ga_kwargs = {'NP': NP}
bat_kwargs = {'NP': NP}
pso_kwargs = {'NP': NP}
es_kwargs = {}
abc_kwargs = {'NP': NP}
de_kwargs = {'NP': NP}
fa_kwargs = {'NP': NP}
gw_kwargs = {'NP': NP}
hs_kwargs = {'HMS': NP}
fw_kwargs = {'n': NP}

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
