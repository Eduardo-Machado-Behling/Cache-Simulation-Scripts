from __future__ import annotations

import os
import pandas as pd
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import re
from typing import *
import collections


@dataclass
class Report:
    @dataclass
    class Simulation:
        inst_amount: int
        ls_amount: int
        time: float
        speed: float

    @dataclass
    class LD:
        @dataclass
        class Region:
            base: int
            size: int

        text: Region
        data: Region
        stack: Region
        prog_entry: int
        environ_base: int
        target_big_endian: bool

    
    @dataclass
    class Cache:
        accesses: int
        hits: int
        misses: int
        replacements: int
        writebacks: int
        invalidations: int
        miss_rate: float
        repl_rate: float
        wb_rate: float
        inv_rate: float
    
    @dataclass
    class Memory:
        page_count: int
        page_mem: int
        ptab_misses: int
        ptab_accesses: int
        ptab_miss_rate: float

    sim: Simulation
    mem: Memory
    ld: LD
    caches: Dict[str, Cache]
    args: Dict[str,str]

    
    def __init__(self, output: str, args: Dict[str, str]):
        data = re.compile(r"([\w.]+)\s+([xa-fA-FkMGmg0-9\.]+)[\s#]+(.*)")

        report = output[output.index("sim: ** simulation statistics **"):]

        def format(x: str) -> Union[int, float]:
            try:
                if '.' in x:
                    return float(x)
                else:
                    return int(x)
            except ValueError:
                return x

        sim = []
        ld = []
        mem = []
        caches = collections.defaultdict(list)
        for match in data.finditer(report):
            if match.group(1).startswith('ld'):
                keys = ['data', 'text', 'stack']
                if match.group(1) in [f'ld_{i}_base' for i in keys]:
                    ld.append([format(match.group(2))])
                elif match.group(1) in [f'ld_{i}_size' for i in keys]:
                    ld[-1].append(format(match.group(2)))
                else:
                    ld.append(format(match.group(2)))
            elif match.group(1).startswith('mem'):
                mem.append(format(match.group(2)))
            elif match.group(1).startswith('sim'):
                sim.append(format(match.group(2)))
            else:
                caches[match.group(1).split('.')[0]].append(format(match.group(2)))
        
        self.sim = Report.Simulation(*sim)
        self.mem = Report.Memory(*mem)
        self.ld = Report.LD(*map(lambda x: Report.LD.Region(*x) if isinstance(x, list) else x, ld))
        self.args = args

        self.caches: dict[str, Cache] = {}
        for k,v in caches.items():
            self.caches[k] = Report.Cache(*v)
    
    def to_df(self, df: Union[None, pd.DataFrame] = None) -> pd.DataFrame:
        keys = collections.defaultdict(list)

        for k,v in self.sim.__dict__.items():
            keys[f"simluation {k.replace('_', ' ')}"].append(v)
        for k,v in self.mem.__dict__.items():
            keys[f"memory {k.replace('_', ' ')}"].append(v)
        for k,v in self.ld.__dict__.items():
            if isinstance(v, Report.LD.Region):
                for k2, v2 in v.__dict__.items():
                    keys[f"ld {k.replace('_', ' ')} {k2.replace('_', ' ')}"].append(v2)
            else:
                keys[f"ld {k.replace('_', ' ')}"].append(v)
        for k,v in self.caches.items():
            for k1, v1 in v.__dict__.items():
                keys[f"cache {k}: {k1.replace('_', ' ')}"].append(v1)

        data = pd.DataFrame({**{k:[v] for k,v in self.args.items()}, **keys})
        return pd.concat(df, data) if df else data



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
        return f"-cache:il{self.level} dl{self.level} -cache:dl{self.level} {super.__str__().split(' ')[1]}"

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
    l1: Config.Cache = NoneCache(1)
    l2: Config.Cache = NoneCache(2)

    def run(self) -> Report:
        args = {}

        cmd = f"sim-cache {self.l1} {self.l2} -tlb:dtlb none -tlb:itlb none {self.bench}".split(' ')
        for k,v in zip(cmd[1:-2:2], cmd[2:-2:2]):
            args[k] = v
        args['benchmark'] = ';'.join(map(lambda x: os.path.basename(x), self.bench.split()))


        print(' '.join(cmd))
        res = sp.run(cmd,  capture_output=True, text=True)
        print(res.returncode)

        return Report(res.stderr, args)









def gen_exp_2() -> None:
    df = None
    config  = Config("./Benchmarks/gcc/cc1.ss ./Benchmarks/gcc/gcc.i", UnifiedCache(1, ))
    report = config.run()
    df = report.to_df(df)
    print(df)
    



def main() -> None:
    gen_exp_2()



if __name__ == '__main__':
    main()