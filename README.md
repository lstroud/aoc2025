# Advent of Code 2025

My solutions for [Advent of Code 2025](https://adventofcode.com/2025). Using this year to get better at numpy, scipy, pandas, and PyTorch.

## Puzzles

### Day 1: Circular Dial Navigation
Track a dial rotating left/right on a circular range and count zero crossings.

Figured out a floor division trick to count crossings without iterating. Numpy makes this almost too easy.

[day1/puzzle1.py](day1/puzzle1.py)

---

### Day 2: Repeating Pattern Detection
Find IDs where digits form repeating patterns.

**Part 1:** First half equals second half (`1212` → `12 == 12`)
**Part 2:** Any repeating pattern - regex `^(.+)\1+$` handles `123123`, `1111`, etc.

[day2/puzzle21.py](day2/puzzle21.py) | [day2/puzzle22.py](day2/puzzle22.py)

---

### Day 3: Largest N-Digit Selection
Pick N digits (in order) from each number to make the largest value.

**Part 1:** Pandas `cummax` from the right finds the best digit after each position. O(n) and satisfying.
**Part 2:** Compared greedy argmax vs monotonic stack. The stack wins.

[day3/puzzle31.py](day3/puzzle31.py) | [day3/puzzle32.py](day3/puzzle32.py)

---

### Day 4: Warehouse Accessibility
Find items in a grid that aren't fully surrounded.

Scipy `convolve` with a 3x3 kernel counts neighbors in one shot. No loops needed.

[day4/puzzle41.py](day4/puzzle41.py) | [day4/puzzle42.py](day4/puzzle42.py)

---

### Day 5: Interval Matching
Find which values fall within which intervals. Sounds simple until there are millions of both.

Went overboard and compared 5 different algorithms:
- **Broadcasting** - Simple O(n×m) matrix. Works until you run out of RAM.
- **Interval Tree** - O(log m) queries. The textbook answer.
- **Sweep Line** - Event-based. Elegant but the constant factors hurt.
- **Pandas IntervalIndex** - Built-in binary search. Surprisingly good.
- **Pandas Native** - `get_indexer_non_unique`. The actual winner.

Benchmarked to find where each strategy wins. Turns out the boring built-in is usually best.

[day5/puzzle52.py](day5/puzzle52.py) | [day5/strategies/](day5/strategies/)

---

### Day 6: Vertical Digit Pivot
Numbers are encoded vertically in columns. Whitespace matters. Read right-to-left, apply operators, sum.

The parsing was the hard part. Ended up comparing loop vs vectorized:
- **Loop** - Iterate columns, filter spaces, join. Clear and correct.
- **Vectorized** - `view(np.uint32)` trick for char-to-int, cumsum for place values. 23× faster but uses 7× more memory.

Sometimes the clever solution isn't worth it. But it was fun to write.

[day6/puzzle62.py](day6/puzzle62.py)

---

### Day 7: Tachyon Beam Splitting
Beams propagate down through a manifold, splitting left/right at `^` splitters.

**Part 1:** Count splits. Beams merge when they converge.
**Part 2:** Count timelines. No merging - every split spawns a parallel universe. Answer is 7+ trillion.

Tried three ways to model this:
- **Matrix Propagation** - Each row is a Markov transition matrix. Linear algebra does the work.
- **Convolution** - Kernel `[1, 0, 1]` spreads signal left and right. Tachyons wait for no one.
- **PyTorch Neural** - Feedforward network framing with hard sigmoid activation. Mostly for fun.

[day7/puzzle71.py](day7/puzzle71.py) | [day7/puzzle72.py](day7/puzzle72.py) | [day7/strategies/](day7/strategies/)

---

### Day 8: Junction Box Circuits
Someone let the elves do electrical work unsupervised. They're wiring up junction boxes floating in 3D space. How do we figure out how many of their circuits we can connect with 1000 wires? 

Tried a few approaches:
- **Hierarchical Clustering** - Single-linkage. Cleanest solution.
- **Union-Find** - Sort pairs by distance, union the closest k. Interesting array structure.
- **Sparse Graph** - Build CSR adjacency, run `connected_components`. BFS does the walking. CSR is interesting.
- **KD-Tree** - `query_pairs` finds pairs within threshold. Still constrained by the distance calc, but by far the fastest.

Extended the puzzle: what if the elves cared about wire length? MST-based wiring saves 15-20% over greedy. 

[day8/puzzle81.py](day8/puzzle81.py) | [day8/puzzle82.py](day8/puzzle82.py) | [day8/strategies/](day8/strategies/)

---

### Day 9: Movie Theater Carpet Fitting
The elves are renovating a movie theater. Their floor plan winds like they were paid by the corner. Find the biggest rectangular carpet that fits inside the renovation zone, corners on red tiles.

Spent too much time implementing vectorized ray casting to prove I didn't need Shapely. I needed Shapely.

Tried three approaches:
- **Shapely Contains** - `polygon.contains(box)`. Works correctly. 
- **Rasterize + Mask** - Convert polygon to boolean grid, use `scipy.ndimage.binary_fill_holes`. Works on small data, memory-explodes when coordinate range hits 100k.
- **Convex Hull + Ray Casting** - Prune candidates to hull vertices, vectorized point-in-polygon with numpy broadcasting. Educational but fundamentally broken for concave polygons.

The hard lesson: **point sampling can't detect edge crossings**.

[day9/puzzle91.py](day9/puzzle91.py) | [day9/puzzle92.py](day9/puzzle92.py) | [day9/strategies/](day9/strategies/)

---

### Day 10: Factory Machine Initialization
A Shiba Inu ate the manual. Configure factory machines by pressing buttons that affect indicator lights or joltage counters. Minimize button presses.

Two completely different mathematical problems hiding behind similar inputs:

**Part 1:** Buttons toggle lights (XOR). Each button pressed 0 or 1 times. This is Gaussian elimination over GF(2) - Galois Field. Gaussian elimination where `1 + 1 = 0`. Free variables? Enumerate all 2^f combinations to find minimum presses.
- **GF(2) Gaussian Elimination** - Same algorithm you learned in linear algebra, but subtraction is addition and division is free. Implemented from scratch numpy doesn't do finite fields.

**Part 2:** Buttons increment counters. Press each button any non-negative integer number of times. Now it's Integer Linear Programming - `scipy.optimize.milp` handles the branch-and-bound. The elves' fingers thank us.
- **scipy.optimize.milp** - LP is polynomial, integers make it NP-hard. Fortunately HiGHS doesn't care about our complexity problems.

The plot twist: Part 1's "simple toggle problem" required implementing Gaussian elimination with mod-2 arithmetic. Part 2's "harder integer problem" requires ILP.

[day10/puzzle101.py](day10/puzzle101.py) | [day10/puzzle102.py](day10/puzzle102.py)

---

### Day 11: Reactor Path Counting
The toroidal reactor's wiring is a mess. Data flows through devices like gossip through the North Pole breakroom - one direction only, but many possible paths.

**Part 1:** Count all paths from `you` to `out`. Classic DFS with memoization. Each device remembers how many paths lead from it to the reactor - because asking the same elf twice is inefficient.

**Part 2:** Count paths from `svr` to `out` that pass through both `dac` (digital-to-analog converter) AND `fft` (fast Fourier transform). Same DFS, but now we track checkpoints. Like Santa checking naughty AND nice - both boxes must be ticked.

The twist: This puzzle was obsurdly easy compared to prior days

[day11/puzzle111.py](day11/puzzle111.py) | [day11/puzzle112.py](day11/puzzle112.py)
