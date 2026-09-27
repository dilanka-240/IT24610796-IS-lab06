# IS Lab 6 - Genetic Algorithm for 0/1 Knapsack Problem
## Student: IT24610796

### Problem
0/1 Knapsack: given n items with weights w[i] and profits v[i], select a subset maximizing total profit without exceeding capacity C.

Dataset: OR-Library P07 (n=15, C=750, optimal=1458)

### GA Design
- Encoding: binary bitstring (1=include item)
- Fitness: hard penalty - value - M * max(0, weight - C), M = 10 * max(v)
- Library: DEAP
- Baseline: tournament k=3, 2-point crossover (pc=0.9), bit-flip mutation (pm=0.02), pop=150, gens=200, seed=42

### Results
Baseline GA achieves 1444 (99.0% of optimal 1458), gap 1.0%.

#### Operator Comparison (vs baseline)
| Variant | Best Fitness | % Baseline |
|---------|-------------|------------|
| Baseline (k=3, 2pt, pm=0.02) | 1444 | 100.0% |
| Tournament k=2 | 1449 | 100.3% |
| Tournament k=4 | 1450 | 100.4% |
| Roulette selection | 1446 | 100.1% |
| Rank selection | 1447 | 100.2% |
| One-point crossover | 1446 | 100.1% |
| Uniform crossover | 1436 | 99.4% |
| pm=0.01 | 1444 | 100.0% |
| pm=0.05 | 1444 | 100.0% |
| pm=0.10 | 1444 | 100.0% |

### Discussion
- Tournament k=4 gave best result (1450) - higher selection pressure helps on this small instance
- Uniform crossover underperformed (1436) - less building-block preservation than 2-point
- Mutation rate changes (0.01 to 0.10) had minimal impact - problem is small enough that baseline pm works well
- All methods close to optimal (within 1-2%)

### How to Run
```bash
cd IS_Lab6
.venv/Scripts/python.exe ga_knapsack.py
```
