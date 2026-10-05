"""
Egyptian merchant catalogue for synthetic transaction generation.

Every merchant here is a real chain trading in Egypt. Amounts are in EGP and are
anchored to publicly known 2024-2025 Egyptian price levels with inflation
applied -- realistic ranges, not quoted prices. Fuel and electricity move with
regulated tariffs, so those two are the ones worth checking against a real bill.

Two fields do the heavy lifting.

`categories`: a merchant with more than one entry is *ambiguous* -- Carrefour
sells groceries and homeware, an Emarat Misr forecourt sells fuel and snacks --
so a transaction there has no single fixed category. That ambiguity is
deliberate; it stops the classifier from memorising a merchant-to-category
lookup table and calling it machine learning.

`weight`: how often the merchant appears. This is where a first attempt went
badly wrong. Giving B.TECH the same weight as a coffee shop had the persona
buying a EGP 4,000 appliance every month, and annual spending came out at 2.2x
income. Frequency has to match how people really shop: coffee and koshary
constantly, a fridge almost never. Amounts are individually realistic; it is the
mix that decides whether the year balances.

A note on `amount`: these are (low, mode, high) triples drawn from a triangular
distribution, whose mean is (low + mode + high) / 3 -- NOT the mode. An
over-generous `high` drags the average far above the typical value, so the tails
are kept tight.
"""

from __future__ import annotations

import random
import re

SPENDING_CATEGORIES = [
    "Groceries",
    "Food",
    "Transport",
    "Shopping",
    "Bills & Utilities",
    "Entertainment",
    "Health & Fitness",
    "Travel",
]

# Cairo districts, used to make descriptors look like a real Egyptian statement.
AREAS = ["MAADI", "ZAMALEK", "NASR CITY", "HELIOPOLIS", "DOKKI", "MOHANDESEEN",
         "NEW CAIRO", "6TH OCTOBER", "GIZA", "SHEIKH ZAYED", "MANIAL"]

MERCHANTS = [
    # ---------------- Groceries ----------------
    {"name": "Carrefour Egypt", "categories": {"Groceries": 0.72, "Shopping": 0.28},
     "amount": (220, 780, 1900), "channels": {"card_present": 0.85, "online": 0.15},
     "style": "pos", "weight": 8},
    {"name": "Kazyon", "categories": {"Groceries": 1.0},
     "amount": (80, 240, 560), "channels": {"card_present": 1.0},
     "style": "pos", "weight": 12},
    {"name": "Seoudi Supermarket", "categories": {"Groceries": 0.94, "Food": 0.06},
     "amount": (160, 480, 1100), "channels": {"card_present": 1.0},
     "style": "visa", "weight": 7},
    {"name": "Spinneys Egypt", "categories": {"Groceries": 0.88, "Shopping": 0.12},
     "amount": (200, 680, 1600), "channels": {"card_present": 1.0},
     "style": "pos", "weight": 4},
    {"name": "Metro Market", "categories": {"Groceries": 1.0},
     "amount": (120, 380, 900), "channels": {"card_present": 1.0},
     "style": "pos", "weight": 8},
    {"name": "Awlad Ragab", "categories": {"Groceries": 1.0},
     "amount": (110, 340, 820), "channels": {"card_present": 1.0},
     "style": "visa", "weight": 7},
    {"name": "Gourmet Egypt", "categories": {"Groceries": 0.80, "Food": 0.20},
     "amount": (260, 760, 1700), "channels": {"card_present": 0.7, "online": 0.3},
     "style": "visa", "weight": 2},

    # ---------------- Food ----------------
    {"name": "Cilantro", "categories": {"Food": 0.96, "Groceries": 0.04},
     "amount": (65, 110, 210), "channels": {"card_present": 0.85, "online": 0.15},
     "style": "visa", "weight": 18},
    {"name": "Beanos Cafe", "categories": {"Food": 1.0},
     "amount": (60, 105, 195), "channels": {"card_present": 1.0},
     "style": "visa", "weight": 9},
    {"name": "TBS The Bakery Shop", "categories": {"Food": 0.85, "Groceries": 0.15},
     "amount": (70, 170, 380), "channels": {"card_present": 0.9, "online": 0.1},
     "style": "pos", "weight": 11},
    {"name": "Koshary Abou Tarek", "categories": {"Food": 1.0},
     "amount": (45, 75, 145), "channels": {"card_present": 1.0},
     "style": "meeza", "weight": 12},
    {"name": "Gad Restaurant", "categories": {"Food": 1.0},
     "amount": (90, 175, 380), "channels": {"card_present": 0.8, "online": 0.2},
     "style": "meeza", "weight": 9},
    {"name": "Zooba", "categories": {"Food": 1.0},
     "amount": (110, 195, 360), "channels": {"card_present": 0.7, "online": 0.3},
     "style": "visa", "weight": 6},
    {"name": "Buffalo Burger", "categories": {"Food": 1.0},
     "amount": (150, 255, 480), "channels": {"card_present": 0.6, "online": 0.4},
     "style": "pos", "weight": 5},
    {"name": "Cook Door", "categories": {"Food": 1.0},
     "amount": (100, 185, 350), "channels": {"card_present": 0.7, "online": 0.3},
     "style": "pos", "weight": 6},
    {"name": "Talabat", "categories": {"Food": 0.92, "Groceries": 0.08},
     "amount": (150, 290, 600), "channels": {"online": 1.0},
     "style": "online", "weight": 10},
    {"name": "Abou El Sid", "categories": {"Food": 1.0},
     "amount": (350, 580, 1100), "channels": {"card_present": 1.0},
     "style": "visa", "weight": 1},

    # ---------------- Transport ----------------
    {"name": "Uber Egypt", "categories": {"Transport": 1.0},
     "amount": (60, 125, 300), "channels": {"online": 1.0},
     "style": "online", "weight": 15},
    {"name": "Careem", "categories": {"Transport": 1.0},
     "amount": (55, 115, 275), "channels": {"online": 1.0},
     "style": "online", "weight": 9},
    {"name": "Cairo Metro", "categories": {"Transport": 1.0},
     "amount": (8, 12, 20), "channels": {"card_present": 1.0},
     "style": "meeza", "weight": 14},
    {"name": "Wataniya Petroleum", "categories": {"Transport": 0.88, "Groceries": 0.12},
     "amount": (400, 780, 1350), "channels": {"card_present": 1.0},
     "style": "pos", "weight": 5},
    {"name": "Misr Petroleum", "categories": {"Transport": 0.90, "Groceries": 0.10},
     "amount": (380, 760, 1300), "channels": {"card_present": 1.0},
     "style": "pos", "weight": 3},
    {"name": "Emarat Misr", "categories": {"Transport": 0.82, "Groceries": 0.18},
     "amount": (350, 720, 1250), "channels": {"card_present": 1.0},
     "style": "pos", "weight": 3},
    {"name": "Swvl", "categories": {"Transport": 1.0},
     "amount": (30, 55, 105), "channels": {"online": 1.0},
     "style": "online", "weight": 6},

    # ---------------- Shopping ----------------
    {"name": "Jumia Egypt", "categories": {"Shopping": 0.80, "Entertainment": 0.11,
                                           "Health & Fitness": 0.09},
     "amount": (120, 380, 1100), "channels": {"online": 1.0},
     "style": "online", "weight": 8},
    {"name": "Noon Egypt", "categories": {"Shopping": 0.82, "Entertainment": 0.10,
                                          "Health & Fitness": 0.08},
     "amount": (140, 420, 1200), "channels": {"online": 1.0},
     "style": "online", "weight": 6},
    # Appliances and laptops: bought once or twice a year, not monthly.
    {"name": "B TECH", "categories": {"Shopping": 0.88, "Entertainment": 0.12},
     "amount": (700, 2600, 9000), "channels": {"card_present": 0.6, "online": 0.4},
     "style": "pos", "weight": 0.5},
    {"name": "2B Computers", "categories": {"Shopping": 1.0},
     "amount": (600, 2200, 7500), "channels": {"card_present": 0.7, "online": 0.3},
     "style": "pos", "weight": 0.5},
    {"name": "Concrete", "categories": {"Shopping": 1.0},
     "amount": (380, 780, 1700), "channels": {"card_present": 0.75, "online": 0.25},
     "style": "visa", "weight": 2},
    {"name": "Town Team", "categories": {"Shopping": 1.0},
     "amount": (330, 700, 1500), "channels": {"card_present": 0.8, "online": 0.2},
     "style": "visa", "weight": 2},
    {"name": "Mobaco Cottons", "categories": {"Shopping": 1.0},
     "amount": (300, 620, 1300), "channels": {"card_present": 0.8, "online": 0.2},
     "style": "visa", "weight": 2},
    {"name": "Home Centre", "categories": {"Shopping": 1.0},
     "amount": (400, 1100, 3200), "channels": {"card_present": 0.8, "online": 0.2},
     "style": "pos", "weight": 0.5},

    # ---------------- Bills & Utilities ----------------
    # weight 0 -> only ever arrive through the recurring-bill schedule
    {"name": "Vodafone Egypt", "categories": {"Bills & Utilities": 1.0},
     "amount": (240, 360, 520), "channels": {"online": 1.0},
     "style": "fawry", "weight": 0},
    {"name": "WE Telecom Egypt", "categories": {"Bills & Utilities": 1.0},
     "amount": (380, 560, 820), "channels": {"online": 1.0},
     "style": "fawry", "weight": 0},
    {"name": "North Cairo Electricity", "categories": {"Bills & Utilities": 1.0},
     "amount": (420, 820, 1500), "channels": {"online": 1.0},
     "style": "fawry", "weight": 0},
    {"name": "Cairo Water Company", "categories": {"Bills & Utilities": 1.0},
     "amount": (95, 155, 260), "channels": {"online": 1.0},
     "style": "fawry", "weight": 0},

    # ---------------- Entertainment ----------------
    {"name": "Netflix Egypt", "categories": {"Entertainment": 1.0},
     "amount": (165, 165, 245), "channels": {"online": 1.0},
     "style": "online", "weight": 0},
    {"name": "Anghami", "categories": {"Entertainment": 1.0},
     "amount": (59.99, 59.99, 89.99), "channels": {"online": 1.0},
     "style": "online", "weight": 0},
    {"name": "Shahid VIP", "categories": {"Entertainment": 1.0},
     "amount": (110, 130, 180), "channels": {"online": 1.0},
     "style": "online", "weight": 0},
    {"name": "VOX Cinemas", "categories": {"Entertainment": 1.0},
     "amount": (120, 180, 340), "channels": {"card_present": 0.7, "online": 0.3},
     "style": "pos", "weight": 4},
    {"name": "Renaissance Cinema", "categories": {"Entertainment": 1.0},
     "amount": (100, 155, 290), "channels": {"card_present": 0.8, "online": 0.2},
     "style": "pos", "weight": 3},
    {"name": "PlayStation Store", "categories": {"Entertainment": 1.0},
     "amount": (200, 480, 1300), "channels": {"online": 1.0},
     "style": "online", "weight": 2},

    # ---------------- Health & Fitness ----------------
    {"name": "El Ezaby Pharmacy", "categories": {"Health & Fitness": 0.66,
                                                 "Groceries": 0.22,
                                                 "Shopping": 0.12},
     "amount": (100, 330, 950), "channels": {"card_present": 0.85, "online": 0.15},
     "style": "pos", "weight": 7},
    {"name": "Seif Pharmacy", "categories": {"Health & Fitness": 0.68,
                                             "Groceries": 0.20,
                                             "Shopping": 0.12},
     "amount": (90, 290, 850), "channels": {"card_present": 1.0},
     "style": "pos", "weight": 5},
    {"name": "Golds Gym Egypt", "categories": {"Health & Fitness": 1.0},
     "amount": (1500, 1750, 2200), "channels": {"online": 1.0},
     "style": "recurring", "weight": 0},
    {"name": "Al Borg Laboratories", "categories": {"Health & Fitness": 1.0},
     "amount": (300, 680, 1700), "channels": {"card_present": 0.7, "online": 0.3},
     "style": "visa", "weight": 1},
    {"name": "Vezeeta Clinic", "categories": {"Health & Fitness": 1.0},
     "amount": (300, 600, 1300), "channels": {"online": 0.6, "card_present": 0.4},
     "style": "online", "weight": 2},

    # ---------------- Travel ----------------
    # Two or three trips a year, so fractional weights. random.choices takes them.
    {"name": "EgyptAir", "categories": {"Travel": 1.0},
     "amount": (2800, 5200, 9500), "channels": {"online": 1.0},
     "style": "online", "weight": 0.5},
    {"name": "Air Cairo", "categories": {"Travel": 1.0},
     "amount": (2200, 3900, 7500), "channels": {"online": 1.0},
     "style": "online", "weight": 0.5},
    {"name": "Go Bus", "categories": {"Travel": 0.85, "Transport": 0.15},
     "amount": (200, 340, 650), "channels": {"online": 0.7, "card_present": 0.3},
     "style": "online", "weight": 2},
    {"name": "Tolip Hotels", "categories": {"Travel": 1.0},
     "amount": (1800, 3400, 7000), "channels": {"card_present": 0.5, "online": 0.5},
     "style": "visa", "weight": 0.5},
]

# Bills that hit on roughly the same day every month.
RECURRING = [
    {"name": "North Cairo Electricity", "day": 6, "jitter": 2},
    {"name": "WE Telecom Egypt", "day": 9, "jitter": 1},
    {"name": "Vodafone Egypt", "day": 13, "jitter": 1},
    {"name": "Cairo Water Company", "day": 19, "jitter": 2},
    {"name": "Netflix Egypt", "day": 15, "jitter": 0},
    {"name": "Anghami", "day": 22, "jitter": 0},
    {"name": "Shahid VIP", "day": 25, "jitter": 0},
    {"name": "Golds Gym Egypt", "day": 3, "jitter": 1},
]

# What a stolen Egyptian card actually gets spent on: electronics that resell
# quickly, airline tickets, and wallet transfers that move money straight out.
# Amounts are large against normal spending but deliberately not ruinous -- a
# card that buys a EGP 200,000 laptop is stopped by a hard limit, not by a model.
# The interesting cases sit close enough to real spending to be arguable.
FRAUD_MERCHANT_POOL = [
    ("B TECH", "Shopping", (2000, 6000)),
    ("2B Computers", "Shopping", (1800, 5500)),
    ("Noon Egypt", "Shopping", (1200, 4000)),
    ("Jumia Egypt", "Shopping", (900, 3000)),
    ("iStore Egypt", "Shopping", (3000, 10000)),
    ("EgyptAir", "Travel", (2500, 8000)),
    ("PlayStation Store", "Entertainment", (600, 2200)),
    ("Vodafone Cash Transfer", "Shopping", (1000, 3500)),
]

MERCHANT_BY_NAME = {m["name"]: m for m in MERCHANTS}


def _mangle(name: str) -> str:
    """
    Bank statements uppercase merchant names and often truncate them to fit a
    fixed-width field -- which is how "TBS The Bakery Shop" becomes
    "TBS THE BAKERY S".
    """
    cleaned = re.sub(r"[^A-Za-z0-9 &]", "", name).upper()
    if len(cleaned) > 17 and random.random() < 0.65:
        cleaned = cleaned[: random.randint(12, 17)].rstrip()
    return cleaned


def _ref(n: int) -> str:
    return "".join(random.choice("0123456789") for _ in range(n))


def make_descriptor(name: str, style: str) -> str:
    """
    Turn a clean merchant name into the noisy string an Egyptian bank prints.

    CIB, NBE and Banque Misr statements carry prefixes that are real signal:
    "POS" for a card terminal, "FAWRY BILL PAY" for a utility, "MEEZA POS" for
    the domestic card scheme, "INSTAPAY TRF" for a wallet transfer. A classifier
    can learn that FAWRY almost always means a bill, which is exactly the sort
    of pattern we want it to discover for itself.
    """
    n = _mangle(name)
    area = random.choice(AREAS)
    if style == "pos":
        return f"POS {_ref(6)} {n} {area} EG"
    if style == "visa":
        return f"VISA PURCHASE {n} {area}"
    if style == "meeza":
        return f"MEEZA POS {n} {_ref(5)}"
    if style == "fawry":
        return f"FAWRY BILL PAY {n} {_ref(9)}"
    if style == "online":
        return f"{n}.COM EG {_ref(6)}"
    if style == "recurring":
        return f"AUTO DEBIT {n} {_ref(6)}"
    if style == "instapay":
        return f"INSTAPAY TRF {n} {_ref(8)}"
    return n


def pick_category(merchant: dict) -> str:
    """Draw one category from the merchant's probability spread."""
    cats = list(merchant["categories"].keys())
    probs = list(merchant["categories"].values())
    return random.choices(cats, weights=probs, k=1)[0]


def pick_channel(merchant: dict) -> str:
    chans = list(merchant["channels"].keys())
    probs = list(merchant["channels"].values())
    return random.choices(chans, weights=probs, k=1)[0]


def draw_amount(merchant: dict) -> float:
    low, mode, high = merchant["amount"]
    return round(random.triangular(low, high, mode), 2)
