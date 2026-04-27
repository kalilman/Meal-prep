# Meal-prep Optimizer Demo

This repository now contains a **fully functional CLI demo** for generating meal-prep plans that balance nutrition, budget, and taste.

## Features

- Ingredient + recipe catalogs loaded from JSON (`data/ingredients.json`, `data/recipes.json`).
- Deterministic optimization using beam search (no random results between runs).
- Objective function combining nutrient coverage, flavor pairings, and budget efficiency.
- Dietary controls:
  - `--vegetarian`
  - `--exclude <ingredient_id>` (repeatable)
  - `--lock <ingredient_id>=<servings>` (repeatable)
- Actionable output:
  - grocery list with quantities and costs
  - nutrient coverage report
  - suggested recipes from selected ingredients
  - prep workflow
- Feedback capture: `--feedback-rating` + `--feedback-note` logged as JSONL.

## Quick start

```bash
python meal_ai.py --people 2 --budget 18
```

## Common demos

Vegetarian plan:

```bash
python meal_ai.py --people 2 --budget 16 --vegetarian
```

Exclude fish and lock rice servings:

```bash
python meal_ai.py --people 3 --budget 28 --exclude salmon --lock brown_rice=3
```

Save feedback after tasting:

```bash
python meal_ai.py --people 1 --budget 10 --feedback-rating 5 --feedback-note "Great flavor and prep speed"
```

## Data model notes

- Nutrient values are per serving and intentionally simple for demo speed.
- Targets are scaled from default daily needs per person.
- This is designed as a demo scaffold; production should ingest USDA FoodData Central and use validated medical guardrails.
