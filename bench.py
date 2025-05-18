from __future__ import annotations

import threading
import sqlite3
import pickle
from pathlib import Path
import tempfile
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
import multiprocessing
from persistqueue import Queue
import itertools


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

    def run(self, log = None) -> Report:
        args = {}

        cmd = f"sim-cache {self.l1} {self.l2} -tlb:dtlb none -tlb:itlb none {self.bench}".split(' ')
        for k, v in zip(cmd[1:-2:2], cmd[2:-2:2]):
            args[k] = v
        args['benchmark'] = ';'.join(
            map(lambda x: os.path.basename(x), self.bench.split()))
        if not isinstance(self.l1, NoneCache):
            args['size'] = self.l1.size
        if not isinstance(self.l2, NoneCache):
            args['size'] += self.l2.size

        if log == None:
            print(' '.join(cmd))
            print("DONE")
        else: 
            log.write(' '.join(cmd) + '\n')
            log.write("DONE\n")
        out = ""
        with tempfile.TemporaryFile(mode='w+') as temp_stderr:
            # Run the subprocess, redirecting stderr to the temporary file
            process = sp.Popen( cmd, stderr=temp_stderr, stdout=sp.DEVNULL, cwd=os.path.join(os.getcwd(), 'benchmarks', 'vortex'))
            process.wait()  # Wait for the subprocess to finish

            temp_stderr.seek(0)
            out = temp_stderr.read()

        return Report(out, args)




BENCHMARKS = [
    f"{os.path.join(os.getcwd(), 'benchmarks', 'go', 'go.ss')} 50 9 {os.path.join(os.getcwd(), 'benchmarks', 'go', '2stone9.in')}",
    f"{os.path.join(os.getcwd(), 'benchmarks', 'vortex', 'vortex.ss')} {os.path.join(os.getcwd(), 'benchmarks', 'vortex', 'tiny.in')}"
]
THRD_AMOUNT = 2

class PersistentQueue:
    def __init__(self, db_path: str):
        self.db_path = Path(f"{db_path}.db")
        self.lock = threading.Lock()
        self.name = db_path
        self._init_db()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # Optional: improve concurrency
        self._cursor = self._conn.cursor()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data BLOB
                )
            """)
            conn.commit()

    def put(self, obj):
        """Put with immediate commit (synchronous)"""
        data = pickle.dumps(obj)
        with self.lock:
            self._conn.execute("INSERT INTO queue (data) VALUES (?)", (data,))
            self._conn.commit()

    def put_unsync(self, obj):
        """Put without commit (deferred sync)"""
        data = pickle.dumps(obj)
        with self.lock:
            self._conn.execute("INSERT INTO queue (data) VALUES (?)", (data,))

    def sync(self):
        """Manually flush pending changes"""
        with self.lock:
            self._conn.commit()

    def get(self):
        with self.lock:
            cursor = self._conn.execute("SELECT id, data FROM queue ORDER BY id LIMIT 1")
            row = cursor.fetchone()
            if row is None:
                return None
            self._conn.execute("DELETE FROM queue WHERE id = ?", (row[0],))
            self._conn.commit()
            return pickle.loads(row[1])

    def empty(self) -> bool:
        with self.lock:
            cursor = self._conn.execute("SELECT 1 FROM queue LIMIT 1")
            return cursor.fetchone() is None

    def size(self) -> int:
        with self.lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM queue")
            return cursor.fetchone()[0]

    def close(self):
        """Ensure all writes are flushed and connection is closed cleanly"""
        with self.lock:
            self._conn.commit()
            self._conn.close()

    def put_many_unsync(self, objs):
        """Buffered insert of many objects without commit."""
        with self.lock:
            self._conn.executemany(
                "INSERT INTO queue (data) VALUES (?)",
                [(pickle.dumps(obj),) for obj in objs]
            )


@dataclass
class Checkpoint:
    df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    inp = PersistentQueue('input')
    out = PersistentQueue('output')

    def load(self) -> bool:
        if os.path.exists("exp_II_check.csv"):
            self.df = pd.read_csv('exp_II_check.csv')
            return True
        else:
            return False

    def save(self):
        print("SAVING CHECKPOINT")
        self.df.to_csv('exp_II_check.csv')
        self.inp.sync()
        self.out.sync()

    def str(self, blocks, sets, ways, benchs) -> str:
        return f"block={blocks[self.block]}, sets={sets[self.sets]}, ways={ways[self.ways]}, bench={benchs[self.benchmark]}"

class RunThread(multiprocessing.Process):
    def __init__(self, queue: str, res: str, group=None, target=None, name=None, args=..., kwargs=None, *, daemon=None):
        super().__init__()
        self.queue = PersistentQueue(queue)
        self.res = PersistentQueue(res)
        self.log = open(f"{self.name}.log", 'w')

    def run(self):
        self.log.write("Running\n")
        while not self.queue.empty():
            self.log.flush()
            task: Config = self.queue.get()
            try:
                self.res.put(task.run())
            except Exception as e:
                print(f"ERROR: {e}\n")
                pass

def gen_exp_2() -> None:
    checkpoint = Checkpoint()

    blocks = [2**i for i in range(3, 30)]
    sets = [2 ** i for i in range(30)]
    ways = [2 ** i for i in range(30)]
    args = list(filter(lambda x: sum(x) <= 30, [(x, y, z) for x in range(3, 30) for y in range(30) for z in range(30)]))

    def populate():
        t = len(args) * 2
        i = 0
        configs = []
        if not os.path.exists('missing.csv'):
            for block, st, way in map(lambda x: (2**x[0], 2**x[1], 2**x[2]), args):
                for bench in BENCHMARKS:
                    config = Config(bench, UnifiedCache(
                        1, st, block, way, 'LRU'))
                    i += 1
                    print(f"[POPULATE {i}/{t}] {config}")
                    configs.append(config)
        else:
            miss = pd.read_csv('missing.csv')
            t = miss.shape[0]
            for row in miss.itertuples(index=False, name='Row'):
                print(row)
                config = Config(BENCHMARKS[row[4]], UnifiedCache(
                    1, row[1], row[3], row[2], 'LRU'))
                i += 1
                print(f"[POPULATE {i}/{t}] {config}")
                checkpoint.inp.put(config)


        checkpoint.inp.put_many_unsync(configs)

    if not checkpoint.load():
        populate()
        checkpoint.save()

    threads = [RunThread(checkpoint.inp.name, checkpoint.out.name)
               for _ in range(THRD_AMOUNT)]


    for thrd in threads:
        thrd.start()

    last = 0
    refresh = 10
    try:
        while not checkpoint.inp.empty() or not checkpoint.out.empty():
            curr = 0
            print("OUT: ", checkpoint.out.size())
            while not checkpoint.out.empty():
                curr += 1
                report: Report = checkpoint.out.get()
                checkpoint.df = report.to_df(checkpoint.df)
            
            rate = (curr - last) / refresh
            remaing = checkpoint.inp.size()
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

def generate_args(max: int):
    for x in range(3, max):
        for y in range(max):
            s1 = x + y
            if s1 > max:
                continue
            for z in range(max):
                s2 = s1 + z
                if s2 > max:
                    continue
                for a in range(3, max):
                    s3 = s2 + a
                    if s3 > max:
                        continue
                    for b in range(max):
                        s4 = s3 + b
                        if s4 > max:
                            continue
                        for c in range(max):
                            if s4 + c <= max:
                                yield (x, y, z, a, b, c)

def gen_args(max: int):
    range_values = range(max)  # Define the range for each variable

    return [
        (x, y, z, a, b, c)
        for x, y, z, a, b, c in itertools.product(range_values, repeat=6)
        if x + y + z + a + b + c <= max and x > 2 and a > 2
    ]

def gen_exp_4() -> None:
    checkpoint = Checkpoint()


    def populate():
        args = list(gen_args(20))
        args = args[len(args)//2:]
        t = len(args) * 2
        i = 0
        configs = []
        for block, st, way, block2, st2, way2 in map(lambda x: map(lambda y: 2**y, x), args):
            for bench in BENCHMARKS:
                config = Config(bench, HavardCache(
                    HavardCache.InstructionCache( 1, st, block, way, 'LRU'),
                    HavardCache.DataCache( 1, st2, block2, way2, 'LRU')
                ))
                i += 1
                print(f"[POPULATE {i}/{t}] {config}")
                configs.append(config)

        checkpoint.inp.put_many_unsync(configs)
        for _ in range(THRD_AMOUNT):
            checkpoint.inp.put(None)

    if not checkpoint.load():
        populate()
        checkpoint.save()

    threads = [RunThread(checkpoint.inp.name, checkpoint.out.name)
               for _ in range(THRD_AMOUNT)]

    for thrd in threads:
        thrd.start()

    last = 0
    refresh = 10
    try:
        while not checkpoint.inp.empty() or not checkpoint.out.empty():
            curr = 0
            print("OUT: ", checkpoint.out.size())
            while not checkpoint.out.empty():
                curr += 1
                report: Report = checkpoint.out.get()
                checkpoint.df = report.to_df(checkpoint.df)
            
            rate = (curr - last) / refresh
            remaing = checkpoint.inp.size()
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

    checkpoint.df.to_csv('exp_I.csv')
    pass


def main() -> None:
    gen_exp_2()


if __name__ == '__main__':
    main()
