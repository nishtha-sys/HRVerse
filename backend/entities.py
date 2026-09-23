"""
HRVerse - Entity Recognition and Employee Lookup (Chapter 3.3, "Entity Recognition").

Example from the report: "How many sick leaves do I have left?"
    Intent  = leave_policy
    Entity  = sick leave

This module does two things for that example:
    1. Decide whether a leave_policy question is a PERSONAL request ("my", "I have", "balance")
       rather than a general policy question ("what is the leave policy").
    2. Pull out WHICH leave type was asked about, and look up the number for that employee
       in hr_data.py (a small sample dataset for this demo, standing in for a real database).
"""

import os
import sys

# hr_data.py lives in the project root, one level above this backend/ folder.
# This works whichever way the server was started (uvicorn backend.main:app,
# or uvicorn main:app from inside backend/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hr_data


PERSONAL_PRONOUNS = {"my", "i", "me", "mine", "myself"}
BALANCE_WORDS = {"balance", "left", "remaining", "remain"}

LEAVE_TYPES = {
    "casual_leave": "casual leaves",
    "sick_leave": "sick leaves",
    "earned_leave": "earned leaves",
}


def is_personal_leave_query(words):
    """`words` should keep stop words (see nlp.normalize), so "my" and "I" are still there."""
    has_leave_word = any(w.startswith("leav") for w in words)
    if not has_leave_word:
        return False
    return bool(words & PERSONAL_PRONOUNS) or bool(words & BALANCE_WORDS)


def extract_leave_type(words):
    """Return which leave type the entity refers to, or None for "all types"."""
    if "sick" in words:
        return "sick_leave"
    if "casual" in words:
        return "casual_leave"
    if "earned" in words or "annual" in words or "paid" in words:
        return "earned_leave"
    return None


def leave_balance_reply(employee_id, leave_type):
    """The response for a personal leave-balance question."""
    if not employee_id:
        return ("To check your leave balance, enter your Employee ID in the sidebar "
                "(try 101 or 102 for this demo), then ask again.")

    employee = hr_data.employees.get(employee_id.strip())
    if not employee:
        return (f"I couldn't find employee ID {employee_id!r}. "
                "This demo only has sample data for IDs 101 and 102.")

    name = employee["name"]
    if leave_type:
        return f"{name}, you have {employee[leave_type]} {LEAVE_TYPES[leave_type]} remaining."

    return (f"{name}, here is your leave balance:\n"
            f"• Casual Leave: {employee['casual_leave']}\n"
            f"• Sick Leave: {employee['sick_leave']}\n"
            f"• Earned Leave: {employee['earned_leave']}")
