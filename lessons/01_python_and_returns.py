# %% [markdown]
# # 1. Python basics, then returns
#
# Start with names, scores and small functions, then use the same operations for returns and fees.
#
# In Jupyter, select a cell and press **Shift+Enter** to run it. Code cells execute Python;
# text cells explain it. Variables live in a **kernel**, the running Python process.
# Use **Kernel → Restart Kernel and Run All Cells** to check that a notebook works from scratch.
#
# ## Values and arithmetic
#
# `=` assigns a value to a name. The annotation after `:` describes its type:
# `str` for text, `int` for a whole number, `float` for a decimal and `bool` for true or false.
# An f-string inserts values inside `{...}`. `:.0%` formats a fraction as a percentage.

# %%
from __future__ import annotations

from math import isclose, isfinite

name: str = "Maya"
completed: int = 3
total: int = 5
score: float = 72.5
finished: bool = completed == total
completion: float = completed / total

print(f"{name} completed {completed} of {total} tasks ({completion:.0%}).")
print("Finished:", finished)
print("Arithmetic:", 3 + 2, 3 * 2, 3 / 2, 3**2)
print("Text:", name.upper(), "  Python  ".strip())

# %% [markdown]
# `==` compares values; it is different from assignment with `=`.
# Type hints help an editor or type checker; Python does not enforce them automatically
# or make the arithmetic faster. `:.2f` displays a number to two decimal places.
# Calling a method such as `name.upper()` asks that object to perform an operation.
#
# ## Lists, dictionaries and tuples
#
# A list holds an ordered collection. Indexing starts at zero; a slice stops **before**
# its end index. A dictionary associates keys with values. A tuple is a fixed sequence.

# %%
scores: list[float] = [60.0, 75.0, 90.0, 45.0]
subjects: dict[str, float] = {"Python": 75.0, "Statistics": 80.0}
profile: tuple[str, int] = ("Maya", 20)

print("First, last, first two:", scores[0], scores[-1], scores[:2])
print("Count, sum, minimum, maximum:", len(scores), sum(scores), min(scores), max(scores))
print("Sorted copy:", sorted(scores))
print("Dictionary lookup:", subjects["Python"])
print("Lookup with a default:", subjects.get("Rust", 0.0))
student, age = profile
print(f"{student} is {age}.")

# %% [markdown]
# Unpacking assigns a tuple's values to separate names.
# Some methods change an object: `append` adds a value to the original list;
# `pop` removes and returns its last value. `sorted` above returned a new list.

# %%
scores.append(80.0)
print("After append:", scores)
removed: float = scores.pop()
print("Removed:", removed, "Remaining:", scores)

# %% [markdown]
# ## Conditions and loops
#
# `if` chooses which instructions run. Indentation groups instructions into a block.
# `for` repeats a block for each value. `enumerate` supplies an index as well as the value.

# %%
if score >= 60.0 and completed > 0:
    print("Passed with at least one task completed.")
else:
    print("More work needed.")

for index, value in enumerate(scores):
    print(f"Score {index}: {value:.1f}")

for subject, mark in subjects.items():
    print(f"{subject}: {mark:.1f}")

# %% [markdown]
# A list comprehension builds a list from a loop, with an optional condition.
# `zip` pairs values from two sequences; `strict=True` checks that their lengths match.

# %%
passing_scores: list[float] = [value for value in scores if value >= 60.0]
print("Passing:", passing_scores)
print("With a five-point bonus:", [value + 5.0 for value in scores])

names: list[str] = ["Maya", "Leo", "Asha", "Sam"]
for student_name, mark in zip(names, scores, strict=True):
    print(f"{student_name}: {mark:.1f}")
assert passing_scores == [60.0, 75.0, 90.0]

# %% [markdown]
# ## Functions and errors
#
# A function gives reusable instructions a name. Parameters describe its inputs;
# `return` sends a result back to the caller. `-> float` annotates the result.
# A parameter can have a default value. `raise` stops an invalid calculation.


# %%
def add_bonus(mark: float, bonus: float = 5.0) -> float:
    return mark + bonus


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list.")
    return sum(values) / len(values)


print("Default bonus:", add_bonus(70.0))
print("Different bonus:", add_bonus(70.0, bonus=10.0))
print("Average:", average(scores))
assert average([10.0, 20.0, 30.0]) == 20.0

# %% [markdown]
# `try`/`except` handles an expected error. An `assert` checks a result while developing;
# validation of function inputs belongs in an explicit check, as in `average`.

# %%
try:
    average([])
except ValueError as error:
    print(f"Rejected input: {error}")

# %% [markdown]
# **Exercise:** write a typed function that keeps values at or above a threshold.
# Use it with `scores` and a threshold of 75. The solution is in the next cell.


# %%
def at_least(values: list[float], threshold: float) -> list[float]:
    return [value for value in values if value >= threshold]


print(at_least(scores, 75.0))
assert at_least(scores, 75.0) == [75.0, 90.0]

# %% [markdown]
# ## Using Python for profit and returns
#
# A long trade buys an asset and later sells it. Buying ten shares at £100 and selling
# at £104 makes £40 before fees. The arithmetic and types work as in the score examples.

# %%
symbol: str = "ALPHA"
quantity: int = 10
entry_price: float = 100.0
exit_price: float = 104.0
is_long: bool = True

gross_profit: float = quantity * (exit_price - entry_price)
print(f"{symbol}: {quantity} shares, gross profit £{gross_profit:.2f}")

# %% [markdown]
# A `float` uses approximate binary arithmetic: suitable for these research examples;
# actual cash ledgers often use integer minor units or decimal arithmetic.

# %%
prices: list[float] = [100.0, 102.0, 101.0, 104.0]
trade: dict[str, float] = {"entry": entry_price, "exit": exit_price}

print("First price:", prices[0])
print("Last price:", prices[-1])
print("First two prices:", prices[:2])
print("Number of observations:", len(prices))
print("Entry stored in a dictionary:", trade["entry"])

# %% [markdown]
# Profit is measured in pounds. A **simple return** is a fraction of the starting price:
# $r = P_{new}/P_{old} - 1$. A return of `0.04` means 4%, not 0.04%.


# %%
def simple_return(previous_price: float, current_price: float) -> float:
    if not (isfinite(previous_price) and isfinite(current_price)):
        raise ValueError("Prices must be finite.")
    if previous_price <= 0.0 or current_price <= 0.0:
        raise ValueError("Prices must be positive.")
    return current_price / previous_price - 1.0


total_return: float = simple_return(prices[0], prices[-1])
print(f"Total price return: {total_return:.2%}")

# %% [markdown]
# `math.isfinite` rejects infinity and NaN (a missing or invalid numerical result).
# Catch the error from an invalid price, then check whether the trade made a profit.

# %%
try:
    simple_return(0.0, 104.0)
except ValueError as error:
    print(f"Rejected input: {error}")

if gross_profit > 0.0:
    print("The long trade made a gross profit.")
else:
    print("The long trade did not make a gross profit.")

# %% [markdown]
# `range(1, len(prices))` gives indices 1, 2, 3.
# For each adjacent pair of prices, we calculate one return and append it to a list.
# Four prices produce **three** returns: we have no earlier price for the first row.

# %%
daily_returns: list[float] = []
for index in range(1, len(prices)):
    daily_returns.append(simple_return(prices[index - 1], prices[index]))

for daily_return in daily_returns:
    print(f"{daily_return:+.2%}")

# %% [markdown]
# Returns **compound**: each day's gain or loss applies to the latest wealth.
# Compound simple returns by multiplying their growth factors, `1 + return`.
# `wealth *= value` is shorthand for `wealth = wealth * value`.

# %%
initial_wealth: float = 1_000.0
wealth: float = initial_wealth
for daily_return in daily_returns:
    wealth *= 1.0 + daily_return

print(f"Final wealth: £{wealth:.2f}")
print(f"Added returns: {sum(daily_returns):.4%}")
print(f"Compounded return: {wealth / initial_wealth - 1.0:.4%}")

assert isclose(wealth, 1_040.0)
assert isclose(wealth / initial_wealth - 1.0, total_return)

# %% [markdown]
# **Exercise:** a broker charges £1 when we buy and £1 when we sell.
# Write a typed function that subtracts both fees from a gross profit.
# Predict the answer for our £40 gross profit before running the solution below.


# %%
def net_profit(gross: float, fee_per_trade: float) -> float:
    if not isfinite(fee_per_trade) or fee_per_trade < 0.0:
        raise ValueError("The fee must be finite and non-negative.")
    return gross - 2.0 * fee_per_trade


profit_after_fees: float = net_profit(gross_profit, 1.0)
print(f"Net profit: £{profit_after_fees:.2f}")
assert isclose(net_profit(40.0, 1.0), 38.0)
