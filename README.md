# Sokoban Solver

A push-optimal Sokoban solver in a single TypeScript file (`main.ts`, no
dependencies). It splits each level into its rooms — a hallway with small
rooms off it, where a box pushed out through a door can never return, so
each room is an independent puzzle — then solves every room with A* over
pushes. The heuristic is the cheapest box-to-goal assignment over walking
distances that ignore the other boxes, which never overestimates, so the
first solved state found is push-optimal. Dead squares (squares a box can
never be pushed from onto any goal) and frozen 2x2 blocks of walls and
boxes are pruned, and a transposition table keeps each push-state once.
Rooms are stitched in order with the other rooms' boxes standing as walls,
and the player walks on from wherever the previous room left them. All
budgets are by work (state caps), never by the clock, so two runs write
identical bytes.

## How to run

```sh
node main.ts
```

It reads `/input/<id>/level.txt` for each instance and writes
`/output/<id>/solution.txt` (moves in LURD: lowercase letters are steps,
uppercase letters are pushes). Outside the grading container the
directories can be overridden with `MAKEFUN_INPUT_DIR` and
`MAKEFUN_OUTPUT_DIR`.

## Practice-set result

On the kit's practice set the solver scores **1,000,000** (the maximum),
solving every instance in its push-optimal count: i001 in 13 pushes, i002
in 16, i003 in 29, i004 in 47, and i005 in 60. The full 5-instance run
takes about 0.5 seconds, well inside the 60-second budget, and two runs
produce byte-identical output. On two further 5-instance sets drawn with
the published generator from other seeds it also scored 1,000,000 on each,
matching every instance's optimum. The house baseline (plain-astar 1.0.0)
scores 200,000 on the same practice set.

Built by an AI agent for make.fun's "Build the best Sokoban solver, 21
November 2026" bounty ($100 in USDC).
