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
    def name(self):
        return self._name

    @staticmethod
    def construct(grammar_name, function_provider, ephemeral_suffix):
        creators = {}
        for subclass in Grammar.__subclasses__():
            creators[subclass._name] = subclass
        return creators[grammar_name](function_provider, ephemeral_suffix)

    def _init_basic_grammar(self):
        print(f"Hi! Build grammar version {self.name} was called. The name of ephc is rsi_overbought_threshold_{self._ephemeral_suffix}")
        pset = gp.PrimitiveSetTyped(f"main-{self.name}", [list], types.FunctionType)
        pset.addPrimitive(operator.lt, [float, float], bool)
        pset.addPrimitive(operator.gt, [float, float], bool)
        pset.addPrimitive(operator.or_, [bool, bool], bool)
        pset.addPrimitive(operator.and_, [bool, bool], bool)
        pset.addPrimitive(self.function_provider.if_then_else, [bool, types.FunctionType, types.FunctionType],
                          types.FunctionType)
        pset.addTerminal(False, bool)
        pset.addTerminal(True, bool)
        pset.addTerminal(self.function_provider.buy, types.FunctionType)
        pset.addTerminal(self.function_provider.sell, types.FunctionType)
        pset.addTerminal(self.function_provider.ignore, types.FunctionType)
        pset.addPrimitive(self.function_provider.identity, [bool], bool, name="identity_bool")
        pset.addPrimitive(self.function_provider.identity, [list], list, name="identity_list")
        pset.addPrimitive(self.function_provider.identity, [float], float, name="identity_float")
        pset.addEphemeralConstant(f"rsi_overbought_threshold_{self._ephemeral_suffix}", lambda: random.uniform(70, 100), float)
        pset.addEphemeralConstant(f"rsi_oversold_threshold_{self._ephemeral_suffix}", lambda: random.uniform(0, 30), float)
        return pset

class GrammarV1(Grammar):

    _name = "gv1"

    def __init__(self, function_provider, ephemeral_suffix=""):
        super(GrammarV1, self).__init__(function_provider)
        self._ephemeral_suffix = ephemeral_suffix  # we need this because Deap stores the grammar in global namespace
                                                   # this creates issues if re-running evolution with unchanged
                                                   # ephemeral constant names!
                                                   # see https://github.com/DEAP/deap/issues/108
        self._build_grammar()


    def _build_grammar(self):
        pset = super()._init_basic_grammar()
        pset.addPrimitive(self.function_provider.rsi, [list], float)
        pset.addPrimitive(self.function_provider.sma50, [list], float)
        pset.addPrimitive(self.function_provider.ema50, [list], float)
        pset.addPrimitive(self.function_provider.sma200, [list], float)
        self._pset = pset

"""
        pset.addTerminal(False, bool)
        pset.addTerminal(True, bool)
        pset.addTerminal(self.function_provider.buy, types.FunctionType)
        pset.addTerminal(self.function_provider.sell, types.FunctionType)
        pset.addTerminal(self.function_provider.ignore, types.FunctionType)
        pset.addPrimitive(self.function_provider.identity, [bool], bool, name="identity_bool")
        pset.addPrimitive(self.function_provider.identity, [list], list, name="identity_list")
        pset.addPrimitive(self.function_provider.identity, [float], float, name="identity_float")
        pset.addEphemeralConstant(f"rsi_overbought_threshold_{self._ephemeral_prefix}", lambda: random.uniform(70, 100), float)
        pset.addEphemeralConstant(f"rsi_oversold_threshold_{self._ephemeral_prefix}", lambda: random.uniform(0, 30), float)
"""




class GrammarV2(Grammar):

    _name = "gv2"

    def __init__(self, function_provider, ephemeral_suffix=""):
        super(GrammarV2, self).__init__(function_provider)
        self._ephemeral_suffix = ephemeral_suffix  # we need this because Deap stores the grammar in global namespace
                                                   # this creates issues if re-running evolution with unchanged
                                                   # ephemeral constant names!
                                                   # see https://github.com/DEAP/deap/issues/108
        self._build_grammar()

    def _build_grammar(self):
        pset = super()._init_basic_grammar()
        pset.addPrimitive(self.function_provider.ema200, [list], float)
        pset.addPrimitive(self.function_provider.price, [list], float)
        self._pset = pset



"""
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
"""