from typing import Any, Dict, List, Tuple
import logging
from fuzzysearch import find_near_matches
from ml_grid.pipeline.data_plot_split import (
    plot_candidate_feature_category_lists,
    plot_dict_values,
)
from ml_grid.util.global_params import global_parameters


def get_pertubation_columns(
    all_df_columns: List[str],
    local_param_dict: Dict[str, Any],
    drop_term_list: List[str],
) -> Tuple[List[str], List[str]]:
    """Categorizes columns and selects features based on configuration.

    This function processes a list of all DataFrame columns, categorizing them
    into groups (e.g., bloods, annotations). It then selects which groups to
    include as features based on boolean flags in `local_param_dict['data']`.
    It also identifies columns to drop based on keywords.

    Args:
        all_df_columns (List[str]): A list of all column names in the DataFrame.
        local_param_dict (Dict[str, Any]): A dictionary of parameters for the
            current run, containing a 'data' sub-dictionary with boolean flags
            for each feature category.
        drop_term_list (List[str]): A list of substrings. Any column name
            containing one of these substrings will be marked for dropping.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - A list of column names selected as features.
            - A list of column names identified to be dropped.
    """
    global_params = global_parameters
    logger = logging.getLogger("ml_grid")
    verbose = global_params.verbose

    # Initial drop list for metadata and unwanted columns
    drop_list = []
    drop_list.extend(
        [
            col
            for col in all_df_columns
            if "__index_level" in col or "Unnamed:" in col or "client_idcode:" in col
        ]
    )

    for drop_term in drop_term_list:
        for elem in all_df_columns:
            if find_near_matches(drop_term, elem.lower(), max_l_dist=0):
                drop_list.append(elem)

    # Define feature categories and their corresponding substrings
    FEATURE_CATEGORIES = {
        "bmi": ["bmi_"],
        "ethnicity": ["census_"],
        "diagnostic_order": [
            "_num-diagnostic-order",
            "_days-since-last-diagnostic-order",
            "_days-between-first-last-diagnostic",
        ],
        "drug_order": [
            "_num-drug-order",
            "_days-since-last-drug-order",
            "_days-between-first-last-drug",
        ],
        "annotation_n": ["_count"],
        "meta_sp_annotation_n": [
            "_count_subject_present",
            "_count_subject_not_present",
            "_count_relative_present",
            "_count_relative_not_present",
        ],
        "annotation_mrc_n": ["_count_mrc_cs"],
        "meta_sp_annotation_mrc_n": [
            "_count_subject_present_mrc_cs",
            "_count_subject_not_present_mrc_cs",
            "_count_relative_present_mrc_cs",
            "_count_relative_not_present_mrc_cs",
        ],
        "core_02": ["core_02_"],
        "bed": ["bed_"],
        "vte_status": ["vte_status_"],
        "hosp_site": ["hosp_site_"],
        "core_resus": ["core_resus_"],
        "news": ["news_resus_"],
        "date_time_stamp": ["date_time_stamp"],
        "appointments": ["ConsultantCode_", "ClinicCode_", "AppointmentType_"],
        # 'bloods' is intentionally last as it's a general catch-all
        "bloods": [
            "_mean",
            "_median",
            "_mode",
            "_std",
            "_num-tests",
            "_days-since-last-test",
            "_max",
            "_min",
            "_most-recent",
            "_earliest-test",
            "_days-between-first-last",
            "_contains-extreme-low",
            "_contains-extreme-high",
            "_basic-obs-feature",
        ],
    }

    categorized_cols = {}
    # Use a set to keep track of columns that have already been assigned to a category
    already_categorized = set()

    for category, substrings in FEATURE_CATEGORIES.items():
        # Find columns that match the substrings but have not yet been categorized
        matches = [
            col
            for col in all_df_columns
            if any(sub in col for sub in substrings) and col not in already_categorized
        ]
        categorized_cols[category] = matches
        # Add the newly found columns to the set of categorized columns
        already_categorized.update(matches)

    if verbose >= 2:
        data = {category: len(cols) for category, cols in categorized_cols.items()}
        plot_candidate_feature_category_lists(data)
    elif verbose >= 1:
        for category, cols in categorized_cols.items():
            logger.info(f"{category}: {len(cols)}")

    pertubation_columns = []
    data_config = local_param_dict.get("data", {})

    # Add explicitly named columns like 'age' and 'sex'
    if data_config.get("age") and "age" in all_df_columns:
        pertubation_columns.append("age")
    if data_config.get("sex") and "male" in all_df_columns:
        pertubation_columns.append("male")

    # Add columns from categories based on the data config toggles
    for category, cols in categorized_cols.items():
        if data_config.get(category):
            pertubation_columns.extend(cols)

    # Add any other columns explicitly set to True in the data dict that were not in a category
    explicitly_selected_cols = {
        col for col, selected in data_config.items() if selected
    }
    for col in explicitly_selected_cols:
        if col not in pertubation_columns and col in all_df_columns:
            pertubation_columns.append(col)

    logger.info(
        f"local_param_dict data perturbation: \n {local_param_dict.get('data')}"
    )

    if verbose >= 2:
        plot_dict_values(local_param_dict.get("data"))

    # Remove duplicates while preserving order
    pertubation_columns = list(dict.fromkeys(pertubation_columns))

    return pertubation_columns, drop_list
