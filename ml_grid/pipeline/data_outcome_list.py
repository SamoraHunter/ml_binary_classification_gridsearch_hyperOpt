from ml_grid.util import outcome_list


def handle_outcome_list(drop_list, outcome_variable):

    print("Extending all outcome list on drop list")

    outcome_object = outcome_list.OutcomeList()

    outcome_list_list = outcome_object.all_outcome_list

    drop_list.extend(outcome_list_list)

    try:
        drop_list.remove(outcome_variable)
    except Exception as e:
        print(outcome_variable + "not in drop list")
        pass

    return drop_list
