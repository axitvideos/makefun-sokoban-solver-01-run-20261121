// Sokoban solver: room decomposition + push-optimal A* per room.
//
// Strategy: a level is a 1-wide hallway with small rooms off it. A box pushed
// out of a room through its door can never come back, so each room is an
// independent puzzle and the level optimum is the sum of the rooms' optima.
// Each room is solved with A* over pushes (one step = one box push), with an
// admissible heuristic (cheapest box-to-goal assignment over walking
// distances that ignore the other boxes), dead-square pruning (a box that can
// never be pushed onto any goal) and frozen 2x2-block pruning. The first
// solved state A* pops is push-optimal. Rooms are stitched in order: the
// other rooms' boxes stand as walls, and the player walks on from wherever
// the previous room left them. All budgets are by work (state caps), never
// by the clock, so two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (states stored), never wall-clock limits.
const ASTAR_STATE_CAP = 600_000;
const BFS_STATE_CAP = 400_000;

// Directions in fixed order: [dx, dy, walk letter, push letter].
const MOVES: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Board {
  width: number;
  height: number;
  solid: Uint8Array; // 1 = wall
  targets: number[]; // goal squares, sorted
  crates: number[]; // box squares, sorted
  worker: number; // player square
}

function readBoard(text: string): Board {
  const lines = text.split('\n').filter((r) => r.length > 0);
  const height = lines.length;
  const width = Math.max(...lines.map((r) => r.length));
  const solid = new Uint8Array(width * height).fill(1);
  const targets: number[] = [];
  const crates: number[] = [];
  let worker = -1;
  lines.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * width + x;
      if (ch === '#') continue;
      solid[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') targets.push(i);
      if (ch === '$' || ch === '*') crates.push(i);
      if (ch === '@' || ch === '+') worker = i;
    }
  });
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (x === 0 || y === 0 || x === width - 1 || y === height - 1) solid[y * width + x] = 1;
    }
  }
  targets.sort((a, b) => a - b);
  crates.sort((a, b) => a - b);
  return { width, height, solid, targets, crates, worker };
}

// Squares a lone box can ever be pushed onto some goal: pull the box back
// from the goals, one step at a time (the player must stand beyond it).
function reachableByBox(board: Board): Uint8Array {
  const { width, solid, targets } = board;
  const live = new Uint8Array(solid.length);
  const queue = [...targets];
  for (const t of targets) live[t] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const cell = queue[qi]!;
    for (const [dx, dy] of MOVES) {
      const d = dy * width + dx;
      const from = cell - d;
      if (!solid[from] && !solid[from - d] && !live[from]) {
        live[from] = 1;
        queue.push(from);
      }
    }
  }
  return live;
}

// Flood-fill the squares the player can walk to from `start` with `blocked`
// (boxes) in the way. Fills `seen` with `tag`, returns the lowest square.
function floodRegion(
  board: Board,
  blocked: Uint8Array,
  start: number,
  seen: Int32Array,
  tag: number,
  scratch: Int32Array,
): number {
  const { width, solid } = board;
  let low = start;
  let n = 0;
  scratch[n++] = start;
  seen[start] = tag;
  while (n > 0) {
    const p = scratch[--n]!;
    if (p < low) low = p;
    const up = p - width;
    const down = p + width;
    const left = p - 1;
    const right = p + 1;
    if (!solid[up] && !blocked[up] && seen[up] !== tag) { seen[up] = tag; scratch[n++] = up; }
    if (!solid[down] && !blocked[down] && seen[down] !== tag) { seen[down] = tag; scratch[n++] = down; }
    if (!solid[left] && !blocked[left] && seen[left] !== tag) { seen[left] = tag; scratch[n++] = left; }
    if (!solid[right] && !blocked[right] && seen[right] !== tag) { seen[right] = tag; scratch[n++] = right; }
  }
  return low;
}

// True when the box just pushed onto `cell` is stuck inside a frozen 2x2
// block of walls and boxes holding a box off its goal.
function isFrozen(board: Board, blocked: Uint8Array, onTarget: Uint8Array, cell: number): boolean {
  const { width, solid } = board;
  const corners = [cell, cell - 1, cell - width, cell - width - 1];
  for (const a of corners) {
    const b = a + 1;
    const c = a + width;
    const d = c + 1;
    if (a < 0 || d >= solid.length) continue;
    const filled = (q: number): boolean => solid[q] === 1 || blocked[q] === 1;
    if (!filled(a) || !filled(b) || !filled(c) || !filled(d)) continue;
    const offGoal = (q: number): boolean => blocked[q] === 1 && onTarget[q] === 0;
    if (offGoal(a) || offGoal(b) || offGoal(c) || offGoal(d)) return true;
  }
  return false;
}

// Walking distances from every square to each goal, ignoring boxes.
// dist[g][cell] = steps, or -1 when unreachable.
function goalDistances(board: Board): Int32Array[] {
  const { width, solid, targets } = board;
  const out: Int32Array[] = [];
  for (const goal of targets) {
    const dist = new Int32Array(solid.length).fill(-1);
    const queue = [goal];
    dist[goal] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of MOVES) {
        const q = p + dy * width + dx;
        if (!solid[q] && dist[q] < 0) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    out.push(dist);
  }
  return out;
}

// All orderings of goal indexes for `n` boxes (n <= 4 here, so <= 24).
function orderings(n: number): number[][] {
  const base = Array.from({ length: n }, (_, i) => i);
  const out: number[][] = [];
  const walk = (prefix: number[], rest: number[]): void => {
    if (rest.length === 0) { out.push(prefix); return; }
    for (let i = 0; i < rest.length; i++) {
      walk([...prefix, rest[i]!], [...rest.slice(0, i), ...rest.slice(i + 1)]);
    }
  };
  walk([], base);
  return out;
}

// Cheapest way to match every box to a distinct goal, over walking
// distances that ignore the other boxes. Admissible: each push moves one
// box one square, so no solution can cost less than this assignment.
function assignmentCost(
  crates: number[],
  dists: Int32Array[],
  perms: number[][],
): number {
  let best = Infinity;
  for (const perm of perms) {
    let sum = 0;
    let ok = true;
    for (let i = 0; i < crates.length; i++) {
      const d = dists[perm[i]!]![crates[i]!]!;
      if (d < 0) { ok = false; break; }
      sum += d;
    }
    if (ok && sum < best) best = sum;
  }
  return best;
}

// Shortest walk from `from` to `to` around walls and boxes (lowercase
// letters, first direction first on ties). Throws when unreachable.
function shortestWalk(board: Board, blocked: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const { width, solid } = board;
  const prev = new Int32Array(solid.length).fill(-1);
  const step = new Array<string>(solid.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of MOVES) {
      const q = p + dy * width + dx;
      if (solid[q] || blocked[q] || prev[q] !== -1) continue;
      prev[q] = p;
      step[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('shortestWalk: destination out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = step[p]! + s;
  return s;
}

interface SearchResult {
  pushes: number;
  moves: string;
}

// Push search over one board: A* when `useHeuristic` is set, plain FIFO
// breadth-first otherwise. Both are push-optimal; A* just gets there with
// far fewer states. Returns null when the state cap runs out or no solution.
function pushSearch(board: Board, cap: number, useHeuristic: boolean): SearchResult | null {
  const { width, solid, targets } = board;
  const size = solid.length;
  const onTarget = new Uint8Array(size);
  for (const t of targets) onTarget[t] = 1;
  const live = reachableByBox(board);
  if (board.crates.some((b) => !live[b])) return null;
  const dists = goalDistances(board);
  const perms = orderings(targets.length);
  const solved = (crates: number[]): boolean => crates.every((b) => onTarget[b] === 1);

  const blocked = new Uint8Array(size);
  const seenMark = new Int32Array(size);
  const regionMark = new Int32Array(size);
  const scratch = new Int32Array(size);
  let tag = 0;

  const keyOf = (crates: number[], low: number): string => low + '|' + crates.join(',');

  // Stored states: crates, player square, parent index, and the push that
  // produced this state (box square before, direction index).
  const stateCrates: number[][] = [board.crates];
  const statePlayer: number[] = [board.worker];
  const stateParent: number[] = [-1];
  const statePushBox: number[] = [-1];
  const statePushDir: number[] = [-1];
  const stateDepth: number[] = [0];
  const table = new Map<string, number>();
  for (const b of board.crates) blocked[b] = 1;
  const startLow = floodRegion(board, blocked, board.worker, seenMark, ++tag, scratch);
  for (const b of board.crates) blocked[b] = 0;
  table.set(keyOf(board.crates, startLow), 0);

  // Open list: a binary heap on [f, h, sequence] for A*, a FIFO queue for BFS.
  const heap: [number, number, number, number][] = []; // [f, h, seq, state]
  let seq = 0;
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      const t = heap[i]!; heap[i] = heap[p]!; heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        const t = heap[i]!; heap[i] = heap[c]!; heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  const fifo: number[] = [0];
  let fifoHead = 0;

  const startH = useHeuristic ? assignmentCost(board.crates, dists, perms) : 0;
  if (solved(board.crates)) return { pushes: 0, moves: '' };
  if (useHeuristic) heapPush([startH, startH, seq++, 0]);

  let goalState = -1;
  while (useHeuristic ? heap.length > 0 : fifoHead < fifo.length) {
    const cur = useHeuristic ? heapPop()[3]! : fifo[fifoHead++]!;
    const crates = stateCrates[cur]!;
    const depth = stateDepth[cur]!;
    for (const b of crates) blocked[b] = 1;
    floodRegion(board, blocked, statePlayer[cur]!, regionMark, ++tag, scratch);
    const here = tag;
    for (let bi = 0; bi < crates.length && goalState < 0; bi++) {
      const box = crates[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = MOVES[di]!;
        const d = dy * width + dx;
        const dest = box + d;
        if (regionMark[box - d] !== here || solid[dest] || blocked[dest] || !live[dest]) continue;
        blocked[box] = 0;
        blocked[dest] = 1;
        if (isFrozen(board, blocked, onTarget, dest)) {
          blocked[dest] = 0;
          blocked[box] = 1;
          continue;
        }
        const next = crates.slice();
        next[bi] = dest;
        next.sort((a, b) => a - b);
        const low = floodRegion(board, blocked, box, seenMark, ++tag, scratch);
        const key = keyOf(next, low);
        blocked[dest] = 0;
        blocked[box] = 1;
        if (table.has(key)) continue;
        if (stateCrates.length >= cap) {
          for (const x of crates) blocked[x] = 0;
          return null;
        }
        const idx = stateCrates.length;
        table.set(key, idx);
        stateCrates.push(next);
        statePlayer.push(box);
        stateParent.push(cur);
        statePushBox.push(box);
        statePushDir.push(di);
        stateDepth.push(depth + 1);
        if (solved(next)) { goalState = idx; break; }
        if (useHeuristic) {
          const h = assignmentCost(next, dists, perms);
          heapPush([depth + 1 + h, h, seq++, idx]);
        } else {
          fifo.push(idx);
        }
      }
    }
    for (const x of crates) blocked[x] = 0;
  }
  if (goalState < 0) return null;

  // Rebuild the move string: walk to each push square, then push.
  const chain: number[] = [];
  for (let s = goalState; s > 0; s = stateParent[s]!) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = board.worker;
  let crates = board.crates;
  for (const s of chain) {
    const nextCrates = stateCrates[s]!;
    const from = crates.find((b) => !nextCrates.includes(b))!;
    const to = nextCrates.find((b) => !crates.includes(b))!;
    const di = statePushDir[s]!;
    const [dx, dy, , pushLetter] = MOVES[di]!;
    const d = dy * width + dx;
    for (const b of crates) blocked[b] = 1;
    moves += shortestWalk(board, blocked, player, from - d);
    for (const b of crates) blocked[b] = 0;
    moves += pushLetter;
    player = from;
    crates = nextCrates;
    void to;
  }
  return { pushes: chain.length, moves };
}

// The player's square after playing `moves` from `from` (walk and push
// steps alike move the player one square).
function endSquare(width: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -width : ch === 'd' ? width : ch === 'l' ? -1 : 1;
  }
  return p;
}

interface Chamber {
  crates: number[];
  targets: number[];
  done: boolean;
}

// Split the level into rooms: cut the hallway row's dead squares (floor
// squares from which no box can reach a goal) out of the floor; each
// remaining floor component holding boxes is a room. Null when the split
// does not cleanly partition boxes and goals.
function splitRooms(board: Board): Chamber[] | null {
  const { width, solid, targets, crates, worker } = board;
  const live = reachableByBox(board);
  const hallRow = Math.floor(worker / width);
  const cut = new Uint8Array(solid.length);
  for (let i = hallRow * width; i < (hallRow + 1) * width; i++) {
    if (!solid[i] && !live[i]) cut[i] = 1;
  }
  if (crates.some((b) => cut[b])) return null;
  const part = new Int32Array(solid.length).fill(-1);
  const chambers: Chamber[] = [];
  for (let s = 0; s < solid.length; s++) {
    if (solid[s] || cut[s] || part[s] >= 0) continue;
    const stack = [s];
    part[s] = chambers.length;
    for (let i = 0; i < stack.length; i++) {
      for (const [dx, dy] of MOVES) {
        const q = stack[i]! + dy * width + dx;
        if (!solid[q] && !cut[q] && part[q] < 0) {
          part[q] = chambers.length;
          stack.push(q);
        }
      }
    }
    chambers.push({ crates: [], targets: [], done: false });
  }
  for (const b of crates) chambers[part[b]!]!.crates.push(b);
  for (const g of targets) {
    if (part[g]! < 0) return null;
    chambers[part[g]!]!.targets.push(g);
  }
  if (chambers.some((c) => c.crates.length !== c.targets.length)) return null;
  return chambers.filter((c) => c.crates.length > 0);
}

// Solve one board, A* first, plain push-BFS as the fallback.
function solveBoard(board: Board): SearchResult | null {
  return (
    pushSearch(board, ASTAR_STATE_CAP, true) ??
    pushSearch(board, BFS_STATE_CAP, false)
  );
}

function solveLevel(text: string): string {
  const board = readBoard(text);
  const chambers = splitRooms(board);
  if (chambers === null) {
    return solveBoard(board)?.moves ?? '';
  }
  let player = board.worker;
  let pushes = 0;
  let moves = '';
  for (const chamber of chambers) {
    const solid = board.solid.slice();
    for (const other of chambers) {
      if (other === chamber) continue;
      // The other rooms' boxes stand as walls; a finished room's boxes sit
      // on its goals, so those squares are the walls instead.
      for (const b of other.done ? other.targets : other.crates) solid[b] = 1;
    }
    const sub: Board = {
      width: board.width,
      height: board.height,
      solid,
      targets: [...chamber.targets].sort((a, b) => a - b),
      crates: [...chamber.crates].sort((a, b) => a - b),
      worker: player,
    };
    const found = solveBoard(sub);
    if (found === null) {
      // A room that will not solve alone: try the whole level as one board.
      return solveBoard(board)?.moves ?? '';
    }
    pushes += found.pushes;
    moves += found.moves;
    player = endSquare(board.width, player, found.moves);
    chamber.done = true;
  }
  void pushes;
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
