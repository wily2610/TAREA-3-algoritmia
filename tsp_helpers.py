import math
import random
import itertools


def tour_length(points, tour):
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]])
               for i in range(len(tour)))


def two_opt_move(route, i, k):
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def two_opt(points, tour, max_iter=200_000):
    n = len(tour)
    tour = tour[:]
    best_len = tour_length(points, tour)
    improvements, it, improved = 0, 0, True
    while improved and it < max_iter:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                if k - i < 2:
                    continue
                cand = two_opt_move(tour, i, k)
                L = tour_length(points, cand)
                it += 1
                if L < best_len - 1e-9:
                    tour, best_len = cand, L
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
    return tour, improvements, best_len


def nearest_neighbor(points, start=0):
    n = len(points)
    unvisited = set(range(n)) - {start}
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda c: math.dist(points[last], points[c]))
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def random_cities(n, seed=42):
    rng = random.Random(seed)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]


def brute_force_optimal(points):
    n = len(points)
    best, best_len = None, math.inf
    for perm in itertools.permutations(range(1, n)):
        t = (0,) + perm
        L = tour_length(points, t)
        if L < best_len:
            best, best_len = t, L
    return list(best), best_len