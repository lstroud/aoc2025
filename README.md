# Advent of Code 2025

My solutions for [Advent of Code 2025](https://adventofcode.com/2025) - an annual coding challenge with daily puzzles throughout December.

## Puzzles

### Day 1: Circular Dial Navigation
Track a dial rotating left/right on a circular range and count zero crossings.

**Approach:** Vectorized numpy operations with a clever floor division trick to count crossings without iterating.

[day1/puzzle1.py](day1/puzzle1.py)

---

### Day 2: Repeating Pattern Detection
Find IDs in ranges where digits form repeating patterns.

**Part 1:** First half equals second half (e.g., `1212` ’ `12 == 12`)
**Part 2:** Any repeating pattern using regex `^(.+)\1+$` (e.g., `123123`, `1111`)

[day2/puzzle21.py](day2/puzzle21.py) | [day2/puzzle22.py](day2/puzzle22.py)

---

### Day 3: Largest N-Digit Selection
Select N digits (in order) from each number to form the largest possible value.

**Part 1:** Uses pandas `cummax` from the right to find the best digit after each position in O(n).
**Part 2:** Compares two approaches - greedy argmax vs monotonic stack.

[day3/puzzle31.py](day3/puzzle31.py) | [day3/puzzle32.py](day3/puzzle32.py)

---

### Day 4: Warehouse Accessibility
Find items in a grid that aren't fully surrounded (accessible from edges).

**Approach:** scipy `convolve` with a 3x3 neighbor kernel to count adjacent items in one vectorized operation.

[day4/puzzle41.py](day4/puzzle41.py) | [day4/puzzle42.py](day4/puzzle42.py)

---

### Day 5: Interval Matching
Find values that fall within any of many intervals, and count total coverage.

**Approach:** Strategy pattern comparing 5 algorithms:
- **Broadcasting** - O(n×m) matrix, simple but memory-heavy
- **Interval Tree** - O(log m) queries via balanced tree
- **Sweep Line** - O((n+m) log(n+m)) event-based processing
- **Pandas IntervalIndex** - Native pandas binary search
- **Pandas Native** - `get_indexer_non_unique` for efficient lookups

Includes benchmarking to show crossover points where different strategies win.

[day5/puzzle52.py](day5/puzzle52.py) | [day5/strategies/](day5/strategies/)
