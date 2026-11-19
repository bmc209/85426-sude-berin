#!/usr/bin/env python3
"""Extract provisional food/non-food labels from partner's Moondream object descriptions.

Moondream CSV is NOT a food classifier: it contains object phrases. The output
contains editable food_present and blank food_prominence columns for review.
"""
import argparse
from pathlib import Path
import re
import pandas as pd
from food_analysis_common import REPO_ROOT, OUT_DIR, image_key

# Conservative edible-item words. This is an EXPLORATORY keyword proxy, not
# ground truth and not a food prominence metric. Match word boundaries so
# 'pineapple' is not the same as 'apple', and 'earring' is not 'ear'.
FOOD_WORDS = [
    'food', 'foods', 'meal', 'meals', 'pizza', 'pizzas', 'sandwich', 'sandwiches',
    'burger', 'burgers', 'hamburger', 'hamburgers', 'fries', 'hot dog', 'hot dogs',
    'cake', 'cakes', 'cupcake', 'cupcakes', 'cheesecake', 'icing', 'frosting',
    'ice cream', 'gelato', 'donut', 'donuts', 'doughnut', 'doughnuts',
    'cookie', 'cookies', 'biscuit', 'biscuits', 'chocolate', 'chocolates',
    'candy', 'candies', 'marshmallow', 'marshmallows', 'lollipop', 'popcorn',
    'bread', 'toast', 'bagel', 'bagels', 'croissant', 'pastry', 'pastries',
    'pasta', 'spaghetti', 'noodles', 'ramen', 'soup', 'salad', 'rice',
    'sushi', 'steak', 'sausage', 'sausages', 'bacon', 'meat', 'meats',
    'shrimp', 'prawn', 'prawns', 'lobster', 'crab meat', 'salmon fillet',
    'roasted chicken', 'fried chicken', 'chicken soup', 'fried egg',
    'boiled egg', 'scrambled eggs', 'omelette', 'eggs', 'egg yolk',
    'banana', 'bananas', 'strawberry', 'strawberries', 'grape', 'grapes',
    'pineapple', 'watermelon', 'melon', 'avocado', 'lemon', 'lemons', 'lime',
    'peach', 'peaches', 'pear', 'pears', 'blueberry', 'blueberries',
    'raspberry', 'raspberries', 'apple', 'apples', 'orange', 'oranges',
    'tomato', 'tomatoes', 'carrot', 'carrots', 'broccoli', 'potato', 'potatoes',
    'mushroom', 'mushrooms', 'onion', 'onions', 'garlic', 'lettuce',
    'cucumber', 'cucumbers', 'corn', 'eggplant', 'zucchini',
    'pepperoni', 'bell pepper', 'bell peppers', 'peppers', 'vegetables',
    'fruit', 'fruits', 'nuts', 'peanuts', 'almonds', 'flour', 'sugar',
    'butter', 'cheese', 'yogurt', 'yoghurt', 'honey', 'jam', 'syrup',
    'waffle', 'waffles', 'pancake', 'pancakes', 'muffin', 'muffins',
    'taco', 'tacos', 'burrito', 'burritos', 'dumpling', 'dumplings',
    'pretzel', 'pretzels', 'cereal', 'oatmeal', 'pie', 'pies', 'scone',
]
BEVERAGE_WORDS = [
    'water', 'coffee', 'tea', 'juice', 'milk', 'soda', 'wine', 'beer',
    'cocktail', 'drink', 'drinks', 'smoothie', 'lemonade',
]
# A word in these object names need not mean the edible item is in the image.
BAD_PHRASES = [
    'apple store', 'apple computer', 'apple logo', 'apple watch',
    'candy wrappers', 'butter dish',
    'food processor', 'food truck', 'food packaging', 'food logo',
    'cake topper', 'cake toppers', 'cake stand', 'cake pan',
    'cookie cutter', 'cookie cutters', 'cookie jar', 'cookie box',
    'rice cooker', 'egg carton', 'egg timer',
    'butterfly', 'butterflies', 'candy cane decoration',
    'orange shirt', 'orange dress', 'orange background', 'orange color',
    'orange fabric', 'orange jacket', 'orange lights', 'orange wall',
    'orange object', 'orange traffic', 'orange flower',
    'breadboard', 'pepper shaker', 'salt and pepper shakers',
    'water bottle',  # beverages are flagged, containers alone not edible items
]
REGEX_FOOD = [(w, re.compile(r'(?<!\w)' + re.escape(w) + r'(?!\w)', re.I)) for w in FOOD_WORDS]
REGEX_DRINK = [(w, re.compile(r'(?<!\w)' + re.escape(w) + r'(?!\w)', re.I)) for w in BEVERAGE_WORDS]
REGEX_BAD = [re.compile(re.escape(x), re.I) for x in BAD_PHRASES]


def score_label(description):
    raw = str(description) if pd.notna(description) else ''
    # Remove known misleading phrases *before* extracting candidates.
    cleaned = raw
    bad_hits = []
    for bad, regex in zip(BAD_PHRASES, REGEX_BAD):
        if regex.search(cleaned):
            bad_hits.append(bad)
            cleaned = regex.sub(' ', cleaned)
    matches = [word for word, rx in REGEX_FOOD if rx.search(cleaned)]
    # Color words such as 'orange scooter' are much more common than edible oranges.
    if 'orange' in matches and not re.search(r'\b(fruit|citrus|peel|peels|slice|slices|branch|juicy)\b', cleaned, re.I):
        matches.remove('orange')
    beverages = [word for word, rx in REGEX_DRINK if rx.search(cleaned)]
    # Caveats that often require checking the actual photo.
    ambiguous = [word for word in ('apple', 'orange', 'eggs', 'food', 'corn', 'meat', 'peppers') if word in matches]
    return int(bool(matches)), '; '.join(matches), '; '.join(beverages), '; '.join(bad_hits + ambiguous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=REPO_ROOT / 'moondream_shared_labels.csv')
    parser.add_argument('--out', type=Path, default=OUT_DIR / 'food_label_review.csv')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow overwriting an existing review file (DESTROYS manual edits).')
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f'{args.out} exists. Refusing to overwrite manual review; use --overwrite to reset it.')
    labels = pd.read_csv(args.labels)
    needed = {'image_index', 'image_name', 'label'}
    if not needed.issubset(labels.columns):
        raise SystemExit(f'Expected CSV columns {sorted(needed)}; found {list(labels.columns)}')
    labels['image_key'] = labels['image_name'].map(image_key)
    if labels['image_key'].duplicated().any() or (labels['image_key'] == '').any():
        raise SystemExit('Duplicate/missing normalized image names; cannot safely align trials.')
    parsed = labels['label'].apply(score_label).tolist()
    labels['food_present_auto'] = [p[0] for p in parsed]
    labels['food_present'] = labels['food_present_auto']  # Manually override 0/1 after checking.
    labels['food_prominence'] = ''  # Optional hand-scored 0..3; never inferred from object tags.
    labels['matched_food_terms'] = [p[1] for p in parsed]
    labels['beverage_terms'] = [p[2] for p in parsed]
    labels['ambiguous_terms'] = [p[3] for p in parsed]
    labels['review_priority'] = [
        'high' if p[3] else 'medium' if (p[0] or p[2]) else 'low' for p in parsed
    ]
    labels['review_status'] = 'unreviewed'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out, index=False)
    print(f'Wrote {args.out}')
    print(f'Images: {len(labels)}, provisional food positives: {int(labels.food_present_auto.sum())}, '
          f'non-food: {int((labels.food_present_auto == 0).sum())}')
    print('IMPORTANT: Object descriptions can omit food or include food-related objects. '
          'Check especially high/medium-priority rows and edit food_present as needed.')
    print('Food prominence is deliberately blank; object lists cannot reliably measure food area/salience.')


if __name__ == '__main__':
    main()
