from typing import List
from ml_grid.util import outcome_list


def handle_outcome_list(drop_list: List[str], outcome_variable: str) -> List[str]:
    """
    This function extends the drop_list with all possible outcome
    variables, then removes the specified outcome_variable from the list

    Args:
        drop_list (List[str]): list of columns to drop
        outcome_variable (str): name of outcome variable

    Returns:
        List[str]: updated drop_list
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
