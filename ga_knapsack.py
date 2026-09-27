import random
from deap import base, creator, tools, algorithms
import os

random.seed(42)

with open(os.path.join(os.path.dirname(__file__), 'p07_c.txt')) as f:
    C = int(f.read().strip())
with open(os.path.join(os.path.dirname(__file__), 'p07_w.txt')) as f:
    w = [int(x) for x in f.read().strip().split()]
with open(os.path.join(os.path.dirname(__file__), 'p07_p.txt')) as f:
    v = [int(x) for x in f.read().strip().split()]
n = len(v)
M = 10 * max(v)

def fitness(ind):
    val = sum(g * vi for g, vi in zip(ind, v))
    wt = sum(g * wi for g, wi in zip(ind, w))
    of = max(0, wt - C)
    return (val - M * of,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

tb = base.Toolbox()
tb.register("attr_bool", random.randint, 0, 1)
tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_bool, n=n)
tb.register("population", tools.initRepeat, list, tb.individual)
tb.register("evaluate", fitness)
tb.register("mate", tools.cxTwoPoint)
tb.register("mutate", tools.mutFlipBit, indpb=0.02)
tb.register("select", tools.selTournament, tournsize=3)

def roulette_select(pop, k, shift):
    fits = [ind.fitness.values[0] + shift for ind in pop]
    total = sum(fits)
    selected = []
    for _ in range(k):
        prog = random.random() * total
        c = 0
        for i, f in enumerate(fits):
            c += f
            if c >= prog:
                selected.append(pop[i])
                break
        else:
            selected.append(pop[-1])
    return selected

def rank_select(pop, k):
    ranked = sorted(pop, key=lambda ind: ind.fitness.values[0])
    selected = []
    n_pop = len(ranked)
    for _ in range(k):
        r = random.random()
        idx = int(r * n_pop)
        idx = min(idx, n_pop - 1)
        selected.append(ranked[idx])
    return selected

def run_evolution(pc=0.9, pm=0.02, tourn_k=3, gens=200, pop_size=150, seed=42,
                   cx_type="two_point", sel_type="tournament"):
    random.seed(seed)
    if cx_type == "one_point":
        mate_fn = tools.cxOnePoint
    elif cx_type == "two_point":
        mate_fn = tools.cxTwoPoint
    else:
        mate_fn = lambda a, b: tools.cxUniform(a, b, indpb=0.5)

    pop = tb.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)

    if sel_type == "roulette":
        min_f = min(ind.fitness.values[0] for ind in pop)
        shift = abs(min_f) + 1 if min_f < 0 else 0
        select_fn = lambda p, k: roulette_select(p, k, shift)
    elif sel_type == "rank":
        select_fn = rank_select
    else:
        tb.register("select", tools.selTournament, tournsize=tourn_k)
        select_fn = lambda p, k: tb.select(p, k)

    hof = tools.HallOfFame(1)
    best_hist = []
    avg_hist = []
    for g in range(gens):
        pop = select_fn(pop, len(pop))
        for i in range(0, len(pop) - 1, 2):
            if random.random() < pc:
                mate_fn(pop[i], pop[i+1])
        for ind in pop:
            if random.random() < 1.0:
                tb.mutate(ind)
        for ind in pop:
            ind.fitness.values = tb.evaluate(ind)
        hof.update(pop)
        fits = [ind.fitness.values[0] for ind in pop]
        best_hist.append(max(fits))
        avg_hist.append(sum(fits) / len(fits))
    return hof[0], hof[0].fitness.values[0], best_hist, avg_hist

def calc_weight(sol):
    return sum(b * w[i] for i, b in enumerate(sol))

def calc_value(sol):
    return sum(b * v[i] for i, b in enumerate(sol))

if __name__ == '__main__':
    print("=== BASELINE (k=3, 2pt cx, pm=0.02) ===")
    best, fit, bh, ah = run_evolution()
    print(f"Best fitness: {fit}")
    print(f"Solution:     {''.join(str(b) for b in best)}")
    print(f"Weight:       {calc_weight(best)} / {C}")
    print(f"Value:        {calc_value(best)}")
    print(f"Optimal (DP): 1458")

    base_fit = fit
    results = {"baseline": fit}

    variants = [
        ("Tournament k=2", "tourn_k2", dict(tourn_k=2)),
        ("Tournament k=4", "tourn_k4", dict(tourn_k=4)),
        ("Roulette selection", "roulette", dict(sel_type="roulette")),
        ("Rank selection", "rank", dict(sel_type="rank")),
        ("One-point crossover", "cx_1pt", dict(cx_type="one_point")),
        ("Uniform crossover", "cx_uniform", dict(cx_type="uniform")),
        ("pm=0.01", "pm_0.01", dict(pm=0.01)),
        ("pm=0.05", "pm_0.05", dict(pm=0.05)),
        ("pm=0.10", "pm_0.10", dict(pm=0.10)),
    ]

    for label, key, kwargs in variants:
        _, fit, _, _ = run_evolution(**kwargs)
        results[key] = fit
        print(f"\n=== {label} ===")
        print(f"Best fitness: {fit}  ({(fit/base_fit*100):.1f}% of baseline)")

    print("\n=== FINAL SUMMARY ===")
    print(f"{'Variant':20s} {'Best':>8s} {'% Baseline':>10s}")
    print("-" * 40)
    for name, val in results.items():
        print(f"{name:20s} {val:8.0f} {val/base_fit*100:9.1f}%")
    print(f"\nOptimal (DP): 1458")
    print(f"Baseline GA:  {base_fit:.0f}")
    print(f"Gap:          {(1 - base_fit/1458)*100:.1f}%")
