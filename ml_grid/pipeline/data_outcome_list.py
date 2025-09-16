from typing import List
from ml_grid.util import outcome_list


def handle_outcome_list(drop_list: List[str], outcome_variable: str) -> List[str]:
    """Extends a drop list with all possible outcome variables, then removes the current one.

    This ensures that only the specified outcome variable for the current run is
    used as the target, and all other potential outcome variables are excluded
    from the feature set.

    Args:
        drop_list (List[str]): The list of columns to be dropped.
        outcome_variable (str): The name of the current outcome variable to keep.

    Returns:
        List[str]: The updated list of columns to drop.
    """

    print("Extending all outcome list on drop list")

    outcome_object = outcome_list.OutcomeList()

    outcome_list_list: List[str] = outcome_object.all_outcome_list

    drop_list.extend(outcome_list_list)

    try:
        drop_list.remove(outcome_variable)
    except Exception as e:
        print(outcome_variable + " not in drop list")
        pass

    return drop_list
