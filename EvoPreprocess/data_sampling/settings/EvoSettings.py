"""
Helper class for default setting of some evolutionary and nature inspired NiaPy algorithms.
"""

# Authors: Sašo Karakatič <karakatic@gmail.com>
# License: MIT


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

hs_kwargs = {}

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
