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
