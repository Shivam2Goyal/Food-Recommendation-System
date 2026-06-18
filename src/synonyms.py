# src/synonyms.py

# Map variant names → canonical name
SYNONYMS = {
    "tomatoes": "tomato",
    "tamatar": "tomato",
    "paneer": "paneer",
    "tofu": "paneer",           # close-enough alias (flag, not merge)
    "ghee": "butter",
    "dalda": "butter",
    "capsicum": "bell pepper",
    "shimla mirch": "bell pepper",
    "coriander leaves": "cilantro",
    "dhania": "cilantro",
    "methi": "fenugreek",
    "hari mirch": "green chili",
    "lal mirch": "red chili",
    "aloo": "potato",
    "matar": "peas",
    "dahi": "yogurt",
    "curd": "yogurt",
    "besan": "chickpea flour",
    "gram flour": "chickpea flour",
    "maida": "all-purpose flour",
    "atta": "whole wheat flour",
    "jeera": "cumin",
    "haldi": "turmeric",
    "adrak": "ginger",
    "lahsun": "garlic",
    "patta gobhi": "cabbage",
    "gajar": "carrot",
    "palak": "spinach",
    "saag": "spinach",
}

def normalise(ingredient: str) -> str:
    """Lowercase, strip whitespace, apply synonym map."""
    cleaned = ingredient.strip().lower()
    return SYNONYMS.get(cleaned, cleaned)