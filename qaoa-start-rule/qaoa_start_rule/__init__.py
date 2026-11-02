"""qaoa_start_rule: constant QAOA starts in normalized units plus a budgeted optimization procedure.

A rule, not a trained network. See README.md for evidence, limits and how to regenerate the constants.
"""

from .optimize import StartRuleResult, optimize
from .rule import (CONSTANT_STARTS, LAYERS, RULE_VERSION, bind_parameters, energy_scale, suggest_starts)
from .simulator import QAOASim

__all__ = ["CONSTANT_STARTS", "LAYERS", "RULE_VERSION", "QAOASim", "StartRuleResult", "bind_parameters",
           "energy_scale", "optimize", "suggest_starts"]
__version__ = "0.1.2"          # package version; RULE_VERSION is the version of the constants
