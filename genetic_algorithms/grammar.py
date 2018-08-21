import operator
import random
import types
from deap import gp

from abc import ABC, abstractmethod


class Grammar(ABC):

    def __init__(self, function_provider):
        self.function_provider = function_provider

    @property
    def pset(self):
        return self._pset

    @property
    @abstractmethod
    def name(self):
        pass


class GrammarV1(Grammar):

    def __init__(self, function_provider, ephemeral_prefix=""):
        super(GrammarV1, self).__init__(function_provider)
        self._ephemeral_prefix = ephemeral_prefix  # we need this because Deap stores the grammar in global namespace
                                                   # this creates issues if re-running evolution with unchanged
                                                   # ephemeral constant names!
                                                   # see https://github.com/DEAP/deap/issues/108
        self._build_grammar()

    def name(self):
        return("g_v1")

    def _build_grammar(self):
        #import importlib, deap
        #gp = importlib.reload(deap.gp)

        pset = gp.PrimitiveSetTyped(f"main-{self.name}", [list], types.FunctionType)
        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.gt, [float, float], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(self.function_provider.if_then_else, [bool, types.FunctionType, types.FunctionType], types.FunctionType)
        pset.addPrimitive(self.function_provider.rsi, [list], float)
        pset.addPrimitive(self.function_provider.sma50, [list], float)
        pset.addPrimitive(self.function_provider.ema50, [list], float)
        pset.addPrimitive(self.function_provider.sma200, [list], float)
        pset.addPrimitive(self.function_provider.ema200, [list], float)
        pset.addPrimitive(self.function_provider.price, [list], float)
        pset.addTerminal(False, bool)
        pset.addTerminal(True, bool)
        pset.addTerminal(self.function_provider.buy, types.FunctionType)
        pset.addTerminal(self.function_provider.sell, types.FunctionType)
        pset.addTerminal(self.function_provider.ignore, types.FunctionType)
        pset.addPrimitive(self.function_provider.identity, [bool], bool, name="identity_bool")
        pset.addPrimitive(self.function_provider.identity, [list], list, name="identity_list")
        pset.addPrimitive(self.function_provider.identity, [float], float, name="identity_float")
        pset.addEphemeralConstant(f"rsi_overbought_threshold_{random.randint(0,1000)}", lambda: random.uniform(70, 100), float)
        pset.addEphemeralConstant(f"rsi_oversold_threshold_{random.randint(0,1000)}", lambda: random.uniform(0, 30), float)
        self._pset = pset



class GrammarV2(Grammar):

    def __init__(self, function_provider, ephemeral_prefix=""):
        super(GrammarV2, self).__init__(function_provider)
        self._ephemeral_prefix = ephemeral_prefix  # we need this because Deap stores the grammar in global namespace
                                                   # this creates issues if re-running evolution with unchanged
                                                   # ephemeral constant names!
                                                   # see https://github.com/DEAP/deap/issues/108
        self._build_grammar()

    def name(self):
        return("g_v2")

    def _build_grammar(self):
        #import importlib, deap
        #gp = importlib.reload(deap.gp)

        pset = gp.PrimitiveSetTyped(f"main-{self.name}", [list], types.FunctionType)
        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.gt, [float, float], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(self.function_provider.if_then_else, [bool, types.FunctionType, types.FunctionType], types.FunctionType)
        pset.addPrimitive(self.function_provider.rsi, [list], float)
        pset.addPrimitive(self.function_provider.sma50, [list], float)
        pset.addPrimitive(self.function_provider.ema50, [list], float)
        pset.addPrimitive(self.function_provider.sma200, [list], float)
        pset.addPrimitive(self.function_provider.ema200, [list], float)
        pset.addPrimitive(self.function_provider.price, [list], float)
        pset.addTerminal(False, bool)
        pset.addTerminal(True, bool)
        pset.addTerminal(self.function_provider.buy, types.FunctionType)
        pset.addTerminal(self.function_provider.sell, types.FunctionType)
        pset.addTerminal(self.function_provider.ignore, types.FunctionType)
        pset.addPrimitive(self.function_provider.identity, [bool], bool, name="identity_bool")
        pset.addPrimitive(self.function_provider.identity, [list], list, name="identity_list")
        pset.addPrimitive(self.function_provider.identity, [float], float, name="identity_float")
        #pset.addEphemeralConstant(f"rsi_overbought_threshold_{random.randint(0,1000)}", lambda: random.uniform(70, 100), float)
        #pset.addEphemeralConstant(f"rsi_oversold_threshold_{random.randint(0,1000)}", lambda: random.uniform(0, 30), float)
        self._pset = pset
