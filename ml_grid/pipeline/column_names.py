from fuzzysearch import find_near_matches

from ml_grid.pipeline.data_plot_split import (
    plot_candidate_feature_category_lists,
    plot_dict_values,
    plot_pie_chart_with_counts,
)
from ml_grid.util.global_params import global_parameters


def filter_substring_list(string, substr):
    return [
        str for str in string if any(sub in str for sub in substr) and "bmi" not in str
    ]


def get_pertubation_columns(all_df_columns, local_param_dict, drop_term_list):
    
    """Identifies and categorizes columns for perturbation and dropping based on predefined rules and
    local parameters.

    This function processes a list of all DataFrame columns, categorizing them into various
    groups such as blood tests, diagnostic orders, drug orders, BMI, ethnicity, and
    different types of annotation counts. It also identifies columns to be dropped based
    on specific keywords and prefixes. The selection of columns for 'perturbation'
    is determined by flags within `local_param_dict`.

    Args:
        all_df_columns (list): A list of all column names in the DataFrame.
        local_param_dict (dict): A dictionary containing local parameters, including
            'outcome_var_n' and a 'data' sub-dictionary that specifies which column
            categories to include for perturbation (e.g., 'age', 'sex', 'bmi', 'bloods').
        drop_term_list (list): A list of strings. Any column name containing these
            strings (case-insensitive) will be added to the `drop_list`.

    Returns:
        tuple: A tuple containing two lists:
            - pertubation_columns (list): A list of column names selected for perturbation
              based on the `local_param_dict` settings.
            - drop_list (list): A list of column names identified to be dropped from the
              DataFrame.

    Raises:
        NameError: If `global_parameters` or `find_near_matches` or `filter_substring_list`
                   or `plot_candidate_feature_category_lists` or `plot_dict_values` are not defined.
                   These are assumed to be accessible in the global scope or imported.

    Notes:
        - The function relies on several global or externally defined functions:
          `global_parameters`, `find_near_matches`, `filter_substring_list`,
          `plot_candidate_feature_category_lists`, and `plot_dict_values`.
        - Column categorization is based on hardcoded substrings and prefixes.
        - Overlapping columns are handled by removing elements from `bloods_list` if
          they appear in any other categorized list.
        - Verbose output and plotting are controlled by the `verbose` level from `global_parameters`.
    """

    global_params = global_parameters

    verbose = global_params.verbose

    orignal_feature_names = all_df_columns

    drop_list = []

    index_level_list = list(filter(lambda k: "__index_level" in k, all_df_columns))

    drop_list.extend(index_level_list)

    Unnamed_list = list(filter(lambda k: "Unnamed:" in k, all_df_columns))

    drop_list.extend(Unnamed_list)

    Unnamed_list = list(filter(lambda k: "client_idcode:" in k, all_df_columns))

    drop_list.extend(Unnamed_list)

    outcome_variable = f'outcome_var_{local_param_dict.get("outcome_var_n")}'

    for i in range(0, len(drop_term_list)):

        drop_term_string = drop_term_list[i]

        for elem in all_df_columns:
            res = find_near_matches(drop_term_string, elem.lower(), max_l_dist=0)

            if len(res) > 0:

                drop_list.append(elem)

    blood_test_substrings = [
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
    ]

    diagnostic_test_substrings = [
        "_num-diagnostic-order",
        "_days-since-last-diagnostic-order",
        "_days-between-first-last-diagnostic",
    ]

    drug_order_substrings = [
        "_num-drug-order",
        "_days-since-last-drug-order",
        "_days-between-first-last-drug",
    ]

    annotation_count_list = list(
        filter(
            lambda k: "_count" in k and "_count_subject_present" not in k,
            all_df_columns,
        )
    )

    appointments_substrings = ["ConsultantCode_", "ClinicCode_", "AppointmentType_"]

    meta_sp_annotation_count_list = list(
        filter(lambda k: "_count_subject_present" in k, all_df_columns)
    )

    not_meta_sp_annotation_count_list = list(
        filter(lambda k: "_count_subject_not_present" in k, all_df_columns)
    )

    meta_rp_annotation_count_list = list(
        filter(lambda k: "_count_relative_present" in k, all_df_columns)
    )

    not_meta_rp_annotation_count_list = list(
        filter(lambda k: "_count_relative_not_present" in k, all_df_columns)
    )

    meta_sp_annotation_count_list.extend(not_meta_sp_annotation_count_list)

    meta_sp_annotation_count_list.extend(meta_rp_annotation_count_list)

    meta_sp_annotation_count_list.extend(not_meta_rp_annotation_count_list)

    diagnostic_order_list = []
    diagnostic_list = filter_substring_list(all_df_columns, diagnostic_test_substrings)
    diagnostic_order_list.extend(diagnostic_list)

    drug_order_list = []
    drug_list = filter_substring_list(all_df_columns, drug_order_substrings)
    drug_order_list.extend(drug_list)

    appointments_list = []
    appointments = filter_substring_list(all_df_columns, appointments_substrings)
    appointments_list.extend(appointments)

    bmi_list = list(filter(lambda k: "bmi_" in k, all_df_columns))

    ethnicity_list = list(filter(lambda k: "census_" in k, all_df_columns))

    annotation_mrc_count_list = list(
        filter(lambda k: "_count_mrc_cs" in k, all_df_columns)
    )

    meta_sp_annotation_mrc_count_list = list(
        filter(lambda k: "_count_subject_present_mrc_cs" in k, all_df_columns)
    )

    not_meta_sp_annotation_mrc_count_list = list(
        filter(lambda k: "_count_subject_not_present_mrc_cs" in k, all_df_columns)
    )

    relative_meta_rp_annotation_mrc_count_list = list(
        filter(lambda k: "_count_relative_present_mrc_cs" in k, all_df_columns)
    )

    not_relative_meta_rp_annotation_mrc_count_list = list(
        filter(lambda k: "_count_relative_not_present_mrc_cs" in k, all_df_columns)
    )

    meta_sp_annotation_mrc_count_list.extend(not_meta_sp_annotation_mrc_count_list)

    meta_sp_annotation_mrc_count_list.extend(relative_meta_rp_annotation_mrc_count_list)

    meta_sp_annotation_mrc_count_list.extend(
        not_relative_meta_rp_annotation_mrc_count_list
    )

    core_02_list = list(filter(lambda k: "core_02_" in k, all_df_columns))

    bed_list = list(filter(lambda k: "bed_" in k, all_df_columns))

    vte_status_list = list(filter(lambda k: "vte_status_" in k, all_df_columns))

    hosp_site_list = list(filter(lambda k: "hosp_site_" in k, all_df_columns))

    core_resus_list = list(filter(lambda k: "core_resus_" in k, all_df_columns))

    news_list = list(filter(lambda k: "news_resus_" in k, all_df_columns))

    bloods_list = filter_substring_list(all_df_columns, blood_test_substrings)

    date_time_stamp_list = list(
        filter(lambda k: "date_time_stamp" in k, all_df_columns)
    )
    
    # Combine these into a single conceptual list for overlap check later
    meta_sp_annotation_all_counts = (
        meta_sp_annotation_count_list +
        not_meta_sp_annotation_count_list +
        meta_rp_annotation_count_list +
        not_meta_rp_annotation_count_list
    )
    # Combine these into a single conceptual list for overlap check later
    meta_sp_annotation_mrc_all_counts = (
        meta_sp_annotation_mrc_count_list +
        not_meta_sp_annotation_mrc_count_list +
        relative_meta_rp_annotation_mrc_count_list +
        not_relative_meta_rp_annotation_mrc_count_list
    )
    
    # --- Post-Processing: Remove overlaps from bloods_list ---
    # Create a set of all columns in other categories
    all_other_categorized_cols = set()
    
    # Add all columns from other specific lists to this set
    all_other_categorized_cols.update(annotation_count_list)
    all_other_categorized_cols.update(meta_sp_annotation_all_counts) # Use the combined list
    all_other_categorized_cols.update(diagnostic_order_list)
    all_other_categorized_cols.update(drug_order_list)
    all_other_categorized_cols.update(bmi_list)
    all_other_categorized_cols.update(ethnicity_list)
    all_other_categorized_cols.update(annotation_mrc_count_list)
    all_other_categorized_cols.update(meta_sp_annotation_mrc_all_counts) # Use the combined list
    all_other_categorized_cols.update(core_02_list)
    all_other_categorized_cols.update(bed_list)
    all_other_categorized_cols.update(vte_status_list) 
    all_other_categorized_cols.update(hosp_site_list)
    all_other_categorized_cols.update(core_resus_list)
    all_other_categorized_cols.update(news_list)
    all_other_categorized_cols.update(date_time_stamp_list)
    all_other_categorized_cols.update(appointments_list)

    # Filter bloods_list: keep only elements NOT found in any other category to avoid vte status and others being added to bloods.
    bloods_list = [col for col in bloods_list if col not in all_other_categorized_cols]


    candidate_feature_category_lists = [
        meta_sp_annotation_count_list,
        annotation_count_list,
        diagnostic_order_list,
        drug_order_list,
        bmi_list,
        ethnicity_list,
        annotation_mrc_count_list,
        meta_sp_annotation_mrc_count_list,
        core_02_list,
        bed_list,
        vte_status_list,
        hosp_site_list,
        core_resus_list,
        news_list,
        bloods_list,
        date_time_stamp_list,
        appointments_list
    ]
    if verbose >= 2:

        data = {}

        for i, lst in enumerate(candidate_feature_category_lists, start=1):
            var_name = [name for name, var in locals().items() if var is lst][0]
            data[var_name] = len(lst)

        plot_candidate_feature_category_lists(data)

    elif verbose >= 1:
        for i, lst in enumerate(candidate_feature_category_lists, start=1):
            var_name = [name for name, var in locals().items() if var is lst][0]
            print(f"{var_name}: {len(lst)}")

    pertubation_columns = []

    if local_param_dict.get("data").get("age") == True:
        pertubation_columns.append("age")

    if local_param_dict.get("data").get("sex") == True:
        pertubation_columns.append("male")

    if local_param_dict.get("data").get("bmi") == True:
        pertubation_columns.extend(bmi_list)

    if local_param_dict.get("data").get("ethnicity") == True:
        pertubation_columns.extend(ethnicity_list)

    if local_param_dict.get("data").get("bloods") == True:
        pertubation_columns.extend(bloods_list)

    if local_param_dict.get("data").get("diagnostic_order") == True:
        pertubation_columns.extend(diagnostic_order_list)

    if local_param_dict.get("data").get("drug_order") == True:
        pertubation_columns.extend(drug_order_list)

    if local_param_dict.get("data").get("annotation_n") == True:
        pertubation_columns.extend(annotation_count_list)

    if local_param_dict.get("data").get("meta_sp_annotation_n") == True:
        pertubation_columns.extend(meta_sp_annotation_count_list)

    if local_param_dict.get("data").get("annotation_mrc_n") == True:
        pertubation_columns.extend(annotation_mrc_count_list)

    if local_param_dict.get("data").get("meta_sp_annotation_mrc_n") == True:
        pertubation_columns.extend(meta_sp_annotation_mrc_count_list)

    if local_param_dict.get("data").get("core_02") == True:
        pertubation_columns.extend(core_02_list)

    if local_param_dict.get("data").get("bed") == True:
        pertubation_columns.extend(bed_list)

    if local_param_dict.get("data").get("vte_status") == True:
        pertubation_columns.extend(vte_status_list)

    if local_param_dict.get("data").get("hosp_site") == True:
        pertubation_columns.extend(hosp_site_list)

    if local_param_dict.get("data").get("core_resus") == True:
        pertubation_columns.extend(core_resus_list)

    if local_param_dict.get("data").get("news") == True:
        pertubation_columns.extend(news_list)

    if local_param_dict.get("data").get("date_time_stamp") == True:
        pertubation_columns.extend(date_time_stamp_list)

    if local_param_dict.get("data").get("appointments") == True:
        pertubation_columns.extend(appointments_list)

    print(f"local_param_dict data perturbation: \n {local_param_dict.get('data')}")

    if verbose >= 2:
        plot_dict_values(local_param_dict.get("data"))

    return pertubation_columns, drop_list
