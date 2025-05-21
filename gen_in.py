from __future__ import annotations

import collections
import os
from typing import *
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import random
import sys
import itertools


@dataclass
class Cache:
    type: Literal['unified', 'instruction', 'data']
    level: int
    sets: int
    blocks: int
    associativity: int
    strategy: Literal['random', 'FIFO', 'LRU']

    def __str__(self):
        return f"-cache:{self.type[0]}l{self.level} {self.type[0]}l{self.level}:{self.sets}:{self.blocks}:{self.associativity}:{self.strategy[0].lower()}"

    @property
    def size(self):
        return self.sets * self.blocks * self.associativity


@dataclass
class UnifiedCache(Cache):
    def __init__(self, level: int, sets: int, blocks: int, associativity: int, strategy: Literal['random', 'FIFO', 'LRU']):
        self.type = 'unified'
        self.level = level
        self.sets = sets
        self.blocks = blocks
        self.associativity = associativity
        self.strategy = strategy


    def __str__(self):
        return f"-cache:il{self.level} dl{self.level} -cache:dl{self.level} {super().__str__().split(' ')[1]}"
    


@dataclass
class HavardCache:
    class InstructionCache(Cache):
        def __init__(self, level: int, sets: int, blocks: int, associativity: int, strategy: Literal['random', 'FIFO', 'LRU']):
            self.type = 'instruction'
            self.level = level
            self.sets = sets
            self.blocks = blocks
            self.associativity = associativity
            self.strategy = strategy

    class DataCache(Cache):
        def __init__(self, level: int, sets: int, blocks: int, associativity: int, strategy: Literal['random', 'FIFO', 'LRU']):
            self.type = 'data'
            self.level = level
            self.sets = sets
            self.blocks = blocks
            self.associativity = associativity
            self.strategy = strategy

    inst: InstructionCache
    data: DataCache

    @property
    def size(self):
        return self.inst.size + self.data.size

    def __str__(self):
        return f"{self.inst} {self.data}"


@dataclass
class NoneCache:
    level: int

    def __str__(self):
        return f"-cache:il{self.level} none -cache:dl{self.level} none"


@dataclass
class Config:
    Cache = Union[UnifiedCache, HavardCache, NoneCache]

    bench: str
    l1: Config.Cache = field(default_factory=lambda: NoneCache(1))
    l2: Config.Cache = field(default_factory=lambda: NoneCache(2))

    def __repr__(self) -> str:
     return f"sim-cache {self.l1} {self.l2} -tlb:dtlb none -tlb:itlb none {self.bench}"



BENCHMARKS = [
    f"{os.path.join(os.getcwd(),'benchmarks','go','go.ss')} 50 9 {os.path.join(os.getcwd(), 'benchmarks','go','2stone9.in')}",
    f"{os.path.join(os.getcwd(),'benchmarks','vortex','vortex.ss')} {os.path.join(os.getcwd(), 'benchmarks','vortex','tiny.in')}"
]

def dump(configs: List[Config]):
    sorted(configs, key=lambda x: x.l1.size)
    with open("args.input", 'w') as args:
        args.writelines(map(lambda x: f"{x}\n", configs))

def gen_agg_cols(df: pd.DataFrame) -> pd.DataFrame:
    cache = df['-cache:dl1'].apply(lambda x: x.split(':'))
    df['cache ul1: nsets'] = cache.apply(lambda x: int(x[1])).astype(int)
    df['cache ul1: blocksize'] = cache.apply(lambda x: int(x[2])).astype(int)
    df['cache ul1: associativity'] = cache.apply(lambda x: int(x[3])).astype(int)
    return df

def gen_agg_cols_2(df: pd.DataFrame) -> pd.DataFrame:
    cache = df['-cache:dl1'].apply(lambda x: x.split(':'))
    df['cache dl1: nsets'] = cache.apply(lambda x: int(x[1])).astype(int)
    df['cache dl1: blocksize'] = cache.apply(lambda x: int(x[2])).astype(int)
    df['cache dl1: associativity'] = cache.apply(lambda x: int(x[3])).astype(int)
    cache = df['-cache:il1'].apply(lambda x: x.split(':'))
    df['cache il1: nsets'] = cache.apply(lambda x: int(x[1])).astype(int)
    df['cache il1: blocksize'] = cache.apply(lambda x: int(x[2])).astype(int)
    df['cache il1: associativity'] = cache.apply(lambda x: int(x[3])).astype(int)
    return df

EXP2_MAX = 25
def populate_exp2():
    args = list(filter(lambda x: sum(x) <= EXP2_MAX, [(x, y, z) for x in range(3, EXP2_MAX + 1) for y in range(EXP2_MAX + 1) for z in range(EXP2_MAX)]))
    t = len(args) * 2
    i = 0
    args = args
    configs = []
    configs_dict = collections.defaultdict(list)
    if not os.path.exists('missing.csv'):
        for block, st, way in set(map(lambda x: (2**x[0], 2**x[1], 2**x[2]), args)):
            for bench in BENCHMARKS:
                config = Config(bench, UnifiedCache(
                    1, st, block, way, 'LRU'))
                i += 1
                print(f"[POPULATE {i}/{t}] {config}")
                configs.append(config)

    dump(configs)

BENCHMARK_TO_DF = {
    BENCHMARKS[0]: "go.ss;50;9;2stone9.in",
    BENCHMARKS[1]: "vortex.ss;tiny.in"
}

def gen_args(min: int, max: int):
    range_values = range(max)  # Define the range for each variable

    def is_pow(n):
        return (n & (n -1)) == 0

    def valid(min, max, x,y,z,a,b,c):
        if x < 3 or a < 3:
            return False

        xyz = 2 ** (x + y + z)
        abc = 2 ** (a + b + c)
        xyzabc = xyz + abc
        
        if not is_pow(xyzabc):
            return False

        return xyzabc >= min and xyzabc <= max



    min = 2 ** min
    max = 2 ** max
    pre = [
        (x, y, z, a, b, c)
        for x, y, z, a, b, c in itertools.product(range_values, repeat=6)
        if valid(min, max, x,y,z,a,b,c)
    ]
    print('pre', len(pre))
    return pre


def check_missing_exp2(df: pd.DataFrame):
    args = list(filter(lambda x: sum(x) <= EXP2_MAX, [(x, y, z) for x in range(3, EXP2_MAX + 1) for y in range(EXP2_MAX + 1) for z in range(EXP2_MAX)]))
    t = len(args) * 2
    i = 0
    configs: List[Config] = []
    sep = ["benchmark", "cache ul1: nsets","cache ul1: blocksize","cache ul1: associativity"]
    if not os.path.exists('missing.csv'):
        for block, st, way in set(map(lambda x: (2**x[0], 2**x[1], 2**x[2]), args)):
            for bench in BENCHMARKS:
                config = Config(bench, UnifiedCache(
                    1, st, block, way, 'LRU'))
                i += 1
                print(f"[POPULATE {i}/{t}] {config}")
                configs.append(config)
    

    existing = set(tuple(row[col] for col in sep) for _, row in df.iterrows())
    # Extract existing configs as a set of tuples

    # Filter only configs not in existing
    def f(config: Config):
        key = (BENCHMARK_TO_DF[config.bench], config.l1.sets, config.l1.blocks, config.l1.associativity)
        return key not in existing

    fil = list(filter(f, configs))

    random.shuffle(fil)
    dump(fil)

def check_missing_exp4(df: pd.DataFrame):
    args = gen_args(int(sys.argv[1]), int(sys.argv[2]))
    t = len(args) * 2
    i = 0
    configs: List[Config] = []
    sep = ["benchmark", "cache il1: nsets","cache il1: blocksize","cache il1: associativity", "cache dl1: nsets","cache dl1: blocksize","cache dl1: associativity" ]
    if not os.path.exists('missing.csv'):
        for block, st, way, block2, st2, way2 in map(lambda x: map(lambda y: 2**y, x), args):
            for bench in BENCHMARKS:
                config = Config(bench, HavardCache(
                    HavardCache.InstructionCache( 1, st, block, way, 'LRU'),
                    HavardCache.DataCache( 1, st2, block2, way2, 'LRU')
                ))
                i += 1
                print(f"[POPULATE {i}/{t}] {config}")
                configs.append(config)
    

    existing = set(tuple(row[col] for col in sep) for _, row in df.iterrows())
    # Extract existing configs as a set of tuples

    # Filter only configs not in existing
    def f(config: Config):
        key = (BENCHMARK_TO_DF[config.bench], config.l1.inst.sets, config.l1.inst.blocks, config.l1.inst.associativity, config.l1.data.sets, config.l1.data.blocks, config.l1.data.associativity)
        return key not in existing

    fil = list(filter(f, configs))

    random.shuffle(fil)
    dump(fil)

def populate_exp4():
    args = gen_args(int(sys.argv[1]), int(sys.argv[2]))
    t = len(args) * 2
    i = 0
    args = args
    configs = []
    if not os.path.exists('missing.csv'):
        for block, st, way, block2, st2, way2 in map(lambda x: map(lambda y: 2**y, x), args):
            for bench in BENCHMARKS:
                config = Config(bench, HavardCache(
                    HavardCache.InstructionCache( 1, st, block, way, 'LRU'),
                    HavardCache.DataCache( 1, st2, block2, way2, 'LRU')
                ))
                i += 1
                print(f"[POPULATE {i}/{t}] {config}")
                configs.append(config)

    dump(configs)



def main() -> None:
    if os.path.exists("out.csv"):
        if len(sys.argv) == 1 or sys.argv[1] == 'exp II':
            check_missing_exp2(gen_agg_cols(pd.read_csv("out.csv")))
        else:
            check_missing_exp4(gen_agg_cols_2(pd.read_csv("out.csv")))
    else:
        if len(sys.argv) == 1 or sys.argv[1] == 'exp II':
            populate_exp2()
        else:
            populate_exp4()


if __name__ == '__main__':
    main()
