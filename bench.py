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
import sys
import threading
import queue
import time
from persistqueue import Queue
import pickle


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
    args: Dict[str, str]

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
                caches[match.group(1).split('.')[0]].append(
                    format(match.group(2)))

        self.sim = Report.Simulation(*sim)
        self.mem = Report.Memory(*mem)
        self.ld = Report.LD(
            *map(lambda x: Report.LD.Region(*x) if isinstance(x, list) else x, ld))
        self.args = args

        self.caches: dict[str, Cache] = {}
        for k, v in caches.items():
            self.caches[k] = Report.Cache(*v)

    def to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = collections.defaultdict(list)

        for k, v in self.sim.__dict__.items():
            keys[f"simluation {k.replace('_', ' ')}"].append(v)
        for k, v in self.mem.__dict__.items():
            keys[f"memory {k.replace('_', ' ')}"].append(v)
        for k, v in self.ld.__dict__.items():
            if isinstance(v, Report.LD.Region):
                for k2, v2 in v.__dict__.items():
                    keys[f"ld {k.replace('_', ' ')} {k2.replace('_', ' ')}"].append(
                        v2)
            else:
                keys[f"ld {k.replace('_', ' ')}"].append(v)
        for k, v in self.caches.items():
            for k1, v1 in v.__dict__.items():
                keys[f"cache {k}: {k1.replace('_', ' ')}"].append(v1)

        data = pd.DataFrame({**{k: [v] for k, v in self.args.items()}, **keys})
        return pd.concat([df, data]) if not df.empty else data


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

    def run(self) -> Report:
        args = {}

        cmd = f"sim-cache {self.l1} {self.l2} -tlb:dtlb none -tlb:itlb none {self.bench}".split(
            ' ')
        for k, v in zip(cmd[1:-2:2], cmd[2:-2:2]):
            args[k] = v
        args['benchmark'] = ';'.join(
            map(lambda x: os.path.basename(x), self.bench.split()))
        if not isinstance(self.l1, NoneCache):
            args['size'] = self.l1.blocks * \
                self.l1.associativity * self.l1.sets
        if not isinstance(self.l2, NoneCache):
            args['size'] += self.l2.blocks * \
                self.l2.associativity * self.l2.sets

        print(' '.join(cmd))
        res = sp.run(cmd,  capture_output=True, text=True)
        print("DONE")

        return Report(res.stderr, args)


@dataclass
class Checkpoint:
    df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    inp = Queue('input.queue', chunksize=5000, autosave=False)
    out = Queue('output.queue', autosave=False)

    def load(self) -> bool:
        if os.path.exists("checkpoint.pickle"):
            with open("checkpoint.pickle", 'rb') as check:
                self.df = pickle.load(check)
            return True
        else:
            return False

    def save(self):
        print("SAVING CHECKPOINT")
        with open("checkpoint.pickle", 'wb') as check:
            pickle.dump(self.df, check)

    def str(self, blocks, sets, ways, benchs) -> str:
        return f"block={blocks[self.block]}, sets={sets[self.sets]}, ways={ways[self.ways]}, bench={benchs[self.benchmark]}"


BENCHMARKS = [
    "./benchmarks/go/go.ss 50 9 ./benchmarks/go/2stone9.in",
    "./benchmarks/vortex/vortex.ss ./benchmarks/vortex/tiny.in"
]
THRD_AMOUNT = 4


class RunThread(threading.Thread):
    def __init__(self, queue: queue.Queue, res: queue.Queue, group=None, target=None, name=None, args=..., kwargs=None, *, daemon=None):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.queue = queue
        self.res = res

    def run(self):
        while True:
            task: Union[Config, None] = self.queue.get()
            if task is None:
                break  # Exit signal
            try:
                self.res.put(task.run())
            except Exception as e:
                print("ERROR: ", e)
                pass
            self.queue.task_done()


def gen_exp_2() -> None:
    checkpoint = Checkpoint()

    blocks = [2**i for i in range(3, 30)]
    sets = [2 ** i for i in range(30)]
    ways = [2 ** i for i in range(30)]
    t = len(blocks) * len(sets) * len(ways) * len(BENCHMARKS)

    def populate():

        i = 0
        for block in blocks:
            for st in sets:
                for way in ways:
                    for bench in BENCHMARKS:
                        config = Config(bench, UnifiedCache(
                            1, st, block, way, 'LRU'))
                        i += 1
                        print(f"[POPULATE {i}/{t}] {Config}")
                        checkpoint.inp.put(config)

    if not checkpoint.load():
        populate()
        checkpoint.save()

    threads = [RunThread(checkpoint.inp, checkpoint.out)
               for _ in range(THRD_AMOUNT)]

    for _ in range(len(threads)):
        checkpoint.inp.put(None)

    for thrd in threads:
        thrd.start()

    last = 0
    refresh = 10
    try:
        while not checkpoint.inp.empty() or not checkpoint.out.empty():
            curr = 0
            while not checkpoint.out.empty():
                curr += 1
                report: Report = checkpoint.out.get_nowait()
                checkpoint.df = report.to_df(checkpoint.df)
                checkpoint.out.task_done()
            
            rate = (curr - last) / refresh
            remaing = checkpoint.inp.qsize()
            estimate = time.strftime('%H:%M:%S', time.gmtime((1/rate)*remaing)) if rate != 0 else "inf"
            print(f"[WORKING] tasks_remaining={remaing}, rate={rate:.4f}(task/s), estimate={estimate}, df={checkpoint.df.shape}")
            last = curr
            if curr != 0:
                checkpoint.save()
            time.sleep(refresh)
    except KeyboardInterrupt:
        checkpoint.save()
        sys.exit()
    except Exception as e:
        print("ERROR: ", e)
        pass

    checkpoint.df.to_csv('exp_II.csv')


def main() -> None:
    gen_exp_2()


if __name__ == '__main__':
    main()
