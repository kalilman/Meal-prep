#!/usr/bin/env python3
"""Fully functional meal-prep demo CLI.

Features:
- Loads ingredient and recipe catalogs from JSON files.
- Optimizes servings with a deterministic beam-search solver under budget.
- Balances nutrition coverage, cost, and taste pairings.
- Supports dietary constraints (vegetarian + excluded ingredients + locked servings).
- Emits grocery list, nutrition report, recipe ideas, and prep workflow.
- Captures user feedback into JSONL for future model training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_TARGETS = {
    "protein_g": 100.0,
    "fiber_g": 30.0,
    "calories": 2100.0,
    "iron_mg": 14.0,
    "vitamin_c_mg": 90.0,
}

DEFAULT_WEIGHTS = {
    "nutrition": 0.65,
    "taste": 0.20,
    "cost": 0.15,
}


@dataclass(frozen=True)
class Ingredient:
    id: str
    name: str
    category: str
    vegetarian: bool
    allergens: List[str]
    unit: str
    cost_per_serving: float
    max_servings: int
    nutrients: Dict[str, float]


@dataclass(frozen=True)
class Recipe:
    id: str
    name: str
    tags: List[str]
    ingredients: List[str]
    steps: List[str]


def load_ingredients(path: Path) -> List[Ingredient]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Ingredient(**item) for item in payload]


def load_recipes(path: Path) -> List[Recipe]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Recipe(**item) for item in payload]


def scale_targets(people: int) -> Dict[str, float]:
    return {key: val * people for key, val in DEFAULT_TARGETS.items()}


def parse_lock_values(raw: List[str]) -> Dict[str, int]:
    locks: Dict[str, int] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid --lock value '{item}'. Expected format ingredient_id=servings")
        ingredient, qty = item.split("=", 1)
        locks[ingredient.strip()] = int(qty.strip())
    return locks


def compute_totals(plan: Dict[str, int], ingredients: Dict[str, Ingredient], nutrient_keys: Iterable[str]) -> Tuple[Dict[str, float], float]:
    totals = {key: 0.0 for key in nutrient_keys}
    cost = 0.0
    for ing_id, qty in plan.items():
        if qty <= 0:
            continue
        ing = ingredients[ing_id]
        cost += ing.cost_per_serving * qty
        for key in totals:
            totals[key] += ing.nutrients.get(key, 0.0) * qty
    return totals, cost


def nutrition_coverage(totals: Dict[str, float], targets: Dict[str, float]) -> float:
    score = 0.0
    for key, target in targets.items():
        ratio = totals[key] / target if target else 1.0
        score += min(ratio, 1.25)
    return score / len(targets)


def taste_score(plan: Dict[str, int], flavor_pairs: set[frozenset[str]]) -> float:
    chosen = [ing_id for ing_id, qty in plan.items() if qty > 0]
    if len(chosen) < 2:
        return 0.15
    pairs = list(combinations(chosen, 2))
    if not pairs:
        return 0.15
    hits = sum(1 for a, b in pairs if frozenset((a, b)) in flavor_pairs)
    return hits / len(pairs)


def objective(plan: Dict[str, int], ingredients: Dict[str, Ingredient], targets: Dict[str, float], budget: float, flavor_pairs: set[frozenset[str]]) -> Tuple[float, Dict[str, float], float, float]:
    totals, cost = compute_totals(plan, ingredients, targets.keys())
    if cost > budget:
        return -1e9, totals, cost, 0.0

    n_score = nutrition_coverage(totals, targets)
    t_score = taste_score(plan, flavor_pairs)
    cost_ratio = cost / budget if budget else 1.0
    c_score = max(0.0, 1.0 - cost_ratio)

    weighted = (
        DEFAULT_WEIGHTS["nutrition"] * n_score
        + DEFAULT_WEIGHTS["taste"] * t_score
        + DEFAULT_WEIGHTS["cost"] * c_score
    )
    return weighted, totals, cost, t_score


def beam_search_plan(
    ingredients: List[Ingredient],
    targets: Dict[str, float],
    budget: float,
    flavor_pairs: set[frozenset[str]],
    locks: Dict[str, int],
    beam_width: int,
) -> Dict[str, object]:
    ingredient_map = {i.id: i for i in ingredients}

    base_plan = {i.id: 0 for i in ingredients}
    for ing_id, qty in locks.items():
        if ing_id not in ingredient_map:
            raise ValueError(f"Locked ingredient '{ing_id}' not found in catalog")
        if qty < 0 or qty > ingredient_map[ing_id].max_servings:
            raise ValueError(f"Locked servings for '{ing_id}' must be between 0 and {ingredient_map[ing_id].max_servings}")
        base_plan[ing_id] = qty

    beam: List[Dict[str, int]] = [base_plan]
    for ingredient in ingredients:
        if ingredient.id in locks:
            continue

        candidates: List[Tuple[float, Dict[str, int]]] = []
        for partial in beam:
            for servings in range(ingredient.max_servings + 1):
                nxt = dict(partial)
                nxt[ingredient.id] = servings
                score, _, _, _ = objective(nxt, ingredient_map, targets, budget, flavor_pairs)
                if score < -1e8:
                    continue
                candidates.append((score, nxt))

        candidates.sort(key=lambda item: item[0], reverse=True)
        beam = [plan for _, plan in candidates[:beam_width]]
        if not beam:
            raise RuntimeError("No feasible plan found. Try a higher budget or fewer constraints.")

    best_plan = None
    best_score = -1e9
    best_totals: Dict[str, float] = {}
    best_cost = 0.0
    best_taste = 0.0

    for plan in beam:
        if not any(qty > 0 for qty in plan.values()):
            continue
        score, totals, cost, t_score = objective(plan, ingredient_map, targets, budget, flavor_pairs)
        if score > best_score:
            best_score = score
            best_plan = plan
            best_totals = totals
            best_cost = cost
            best_taste = t_score

    if best_plan is None:
        raise RuntimeError("No valid non-empty plan found.")

    return {
        "score": best_score,
        "plan": {k: v for k, v in best_plan.items() if v > 0},
        "totals": best_totals,
        "cost": best_cost,
        "taste": best_taste,
    }


def choose_recipes(recipes: List[Recipe], selected_ingredients: set[str], vegetarian: bool, max_recipes: int = 3) -> List[Recipe]:
    picks: List[Tuple[int, Recipe]] = []
    for recipe in recipes:
        if vegetarian and "vegetarian" not in recipe.tags:
            continue
        overlap = len(set(recipe.ingredients) & selected_ingredients)
        if overlap == 0:
            continue
        picks.append((overlap, recipe))
    picks.sort(key=lambda x: x[0], reverse=True)
    return [recipe for _, recipe in picks[:max_recipes]]


def prep_workflow(selected_recipes: List[Recipe]) -> List[str]:
    steps = [
        "Wash/chop produce and label meal containers.",
        "Cook batch grains/legumes first, then proteins, then quick vegetables.",
    ]
    for recipe in selected_recipes:
        steps.append(f"Prepare {recipe.name}: {recipe.steps[0]}")
    steps.append("Portion meals evenly, refrigerate up to 4 days, freeze extras.")
    return steps


def append_feedback(path: Path, args: argparse.Namespace, result: Dict[str, object]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "people": args.people,
        "budget": args.budget,
        "vegetarian": args.vegetarian,
        "exclude": args.exclude,
        "lock": args.lock,
        "rating": args.feedback_rating,
        "note": args.feedback_note,
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def build_flavor_pairs() -> set[frozenset[str]]:
    return {
        frozenset(("oats", "banana")),
        frozenset(("black_beans", "brown_rice")),
        frozenset(("chicken_breast", "broccoli")),
        frozenset(("eggs", "spinach")),
        frozenset(("greek_yogurt", "banana")),
        frozenset(("tofu", "broccoli")),
        frozenset(("lentils", "spinach")),
        frozenset(("salmon", "brown_rice")),
    }


def filter_ingredients(ingredients: List[Ingredient], vegetarian: bool, excludes: List[str]) -> List[Ingredient]:
    excluded = set(excludes)
    filtered: List[Ingredient] = []
    for ing in ingredients:
        if ing.id in excluded:
            continue
        if vegetarian and not ing.vegetarian:
            continue
        filtered.append(ing)
    if not filtered:
        raise RuntimeError("No ingredients remain after filters.")
    return filtered


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Meal-prep optimizer demo")
    p.add_argument("--people", type=int, default=1, help="Number of people to feed.")
    p.add_argument("--budget", type=float, default=15.0, help="Daily total budget in USD.")
    p.add_argument("--vegetarian", action="store_true", help="Use vegetarian ingredients/recipes only.")
    p.add_argument("--exclude", action="append", default=[], help="Ingredient id to exclude (repeatable).")
    p.add_argument("--lock", action="append", default=[], help="Lock servings: ingredient_id=servings (repeatable).")
    p.add_argument("--beam-width", type=int, default=120, help="Beam width for optimization search.")
    p.add_argument("--ingredient-file", default="data/ingredients.json", help="Ingredient catalog JSON path.")
    p.add_argument("--recipe-file", default="data/recipes.json", help="Recipe catalog JSON path.")
    p.add_argument("--feedback-rating", type=int, choices=range(1, 6), help="Optional 1-5 meal rating.")
    p.add_argument("--feedback-note", default="", help="Optional free-text feedback note.")
    p.add_argument("--feedback-log", default="data/feedback_log.jsonl", help="Feedback output path.")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.people <= 0:
        raise ValueError("--people must be > 0")
    if args.budget <= 0:
        raise ValueError("--budget must be > 0")
    if args.beam_width < 20:
        raise ValueError("--beam-width should be at least 20")

    ingredients = load_ingredients(Path(args.ingredient_file))
    recipes = load_recipes(Path(args.recipe_file))
    ingredients = filter_ingredients(ingredients, vegetarian=args.vegetarian, excludes=args.exclude)
    locks = parse_lock_values(args.lock)
    targets = scale_targets(args.people)
    flavor_pairs = build_flavor_pairs()

    result = beam_search_plan(
        ingredients=ingredients,
        targets=targets,
        budget=args.budget,
        flavor_pairs=flavor_pairs,
        locks=locks,
        beam_width=args.beam_width,
    )

    ingredient_map = {ing.id: ing for ing in ingredients}
    selected_ids = set(result["plan"].keys())
    recipe_picks = choose_recipes(recipes, selected_ids, vegetarian=args.vegetarian)

    print(f"Meal-prep demo for {args.people} people (budget {format_money(args.budget)}):")
    print("\nGrocery list")
    for ing_id, servings in sorted(result["plan"].items()):
        ing = ingredient_map[ing_id]
        print(f"  - {ing.name}: {servings} x {ing.unit} ({format_money(ing.cost_per_serving * servings)})")

    print("\nNutrition coverage")
    for nutrient, target in targets.items():
        total = result["totals"][nutrient]
        pct = (total / target * 100) if target else 0.0
        print(f"  - {nutrient}: {total:.1f} / {target:.1f} ({pct:.0f}%)")

    print(f"\nEstimated cost: {format_money(result['cost'])}")
    print(f"Taste score: {result['taste']:.2f} (0-1)")
    print(f"Objective score: {result['score']:.3f}")

    print("\nSuggested recipes")
    if recipe_picks:
        for recipe in recipe_picks:
            print(f"  - {recipe.name} [{', '.join(recipe.tags)}]")
    else:
        print("  - No recipe matches found for selected ingredients.")

    print("\nPrep workflow")
    for idx, step in enumerate(prep_workflow(recipe_picks), start=1):
        print(f"  {idx}. {step}")

    if args.feedback_rating:
        append_feedback(Path(args.feedback_log), args, result)
        print(f"\nSaved feedback to {args.feedback_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
