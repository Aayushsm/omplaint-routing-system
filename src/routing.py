"""
routing.py
-----------
This module defines the mapping between predicted complaint categories and
the departments responsible for handling them.

IMPORTANT (Stage 2 of the pipeline):
The Machine Learning model (Stage 1) ONLY predicts a complaint CATEGORY.
It has no concept of departments, org-charts, or business rules.

This module is the separate ROUTING LOGIC (Stage 2) that takes the
predicted category and maps it to a department using a simple lookup
table. Keeping this separate from the ML model keeps the system easy to
explain and easy to change (e.g. if the bank reorganizes departments,
only this file needs to change -- the model does not need retraining).
"""

# Category -> Department mapping (single source of truth)
CATEGORY_TO_DEPARTMENT = {
    "Payment Issue": "Payments Department",
    "Failed Transaction": "Payments Department",
    "Duplicate Transaction": "Payments Department",
    "ATM Issue": "ATM Operations",
    "Card Issue": "Card Services",
    "Login / Account Access": "IT Support",
    "Account Services": "Account Services Department",
    "Charges / Fees": "Finance Department",
    "Fraud / Suspicious Activity": "Fraud & Security",
    "Loan / EMI": "Loans Department",
    "Refund / Reversal": "Payments & Refunds",
    "Mobile Banking / App Issue": "IT Support",
}

# Default status assigned automatically when a complaint is routed
DEFAULT_STATUS = "Automatically Routed"

# Allowed status values (used by the dashboard status-update dropdown)
STATUS_OPTIONS = ["Automatically Routed", "In Progress", "Resolved"]

# Ordered list of all supported complaint categories (used by training +
# UI so the list is defined in exactly one place)
ALL_CATEGORIES = list(CATEGORY_TO_DEPARTMENT.keys())


def route_complaint(category: str) -> str:
    """
    Map a predicted complaint category to the responsible department.

    Parameters
    ----------
    category : str
        The complaint category predicted by the ML model (Stage 1).

    Returns
    -------
    str
        The department the complaint should be routed to (Stage 2).
        Falls back to "General Support" if the category is unrecognised
        (should not normally happen since the model only predicts
        categories it was trained on).
    """
    return CATEGORY_TO_DEPARTMENT.get(category, "General Support")
