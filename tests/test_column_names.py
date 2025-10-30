import unittest

from ml_grid.pipeline.column_names import get_pertubation_columns
from ml_grid.util.global_params import global_parameters


class TestColumnNames(unittest.TestCase):

    def setUp(self):
        """Set up common variables for tests."""
        self.all_df_columns = [
            "age",
            "male",
            "bmi_val",
            "census_A",
            "bloods_test_mean",
            "diag_order_num-diagnostic-order",
            "drug_order_num-drug-order",
            "annotation_1_count",
            "meta_sp_annotation_1_count_subject_present",
            "annotation_mrc_1_count_mrc_cs",
            "meta_sp_annotation_mrc_1_count_subject_present_mrc_cs",
            "core_02_feature",
            "bed_feature",
            "vte_status_feature",
            "hosp_site_A",
            "core_resus_feature",
            "news_resus_feature",
            "date_time_stamp_2022",
            "appointments_ConsultantCode_X",
            "outcome_var_1",
            "some_col__index_level_0",
            "Unnamed: 0",
        ]
        self.drop_term_list = ["bad_term"]
        # Mute verbose output for tests
        global_parameters.verbose = 0

    def test_get_pertubation_columns_selects_all(self):
        """Test that all categories are selected when flags are True."""
        local_param_dict = {
            "outcome_var_n": 1,
            "data": {
                "age": True,
                "sex": True,
                "bmi": True,
                "ethnicity": True,
                "bloods": True,
                "diagnostic_order": True,
                "drug_order": True,
                "annotation_n": True,
                "meta_sp_annotation_n": True,
                "annotation_mrc_n": True,
                "meta_sp_annotation_mrc_n": True,
                "core_02": True,
                "bed": True,
                "vte_status": True,
                "hosp_site": True,
                "core_resus": True,
                "news": True,
                "date_time_stamp": True,
                "appointments": True,
            },
        }
        pert_cols, _ = get_pertubation_columns(
            self.all_df_columns, local_param_dict, self.drop_term_list
        )
        # Expected columns: age, male, bmi_val, census_A, bloods_test_mean,
        # diag_order, drug_order, annotation_1_count, meta_sp_annotation_1,
        # annotation_mrc_1, meta_sp_annotation_mrc_1, core_02_feature,
        # bed_feature, vte_status_feature, hosp_site_A, core_resus_feature,
        # news_resus_feature, date_time_stamp_2022, appointments_ConsultantCode_X
        self.assertEqual(len(pert_cols), 19)

    def test_get_pertubation_columns_selects_none(self):
        """Test that no categories are selected when flags are False."""
        local_param_dict = {
            "outcome_var_n": 1,
            "data": {
                key: False
                for key in [
                    "age",
                    "sex",
                    "bmi",
                    "ethnicity",
                    "bloods",
                    "diagnostic_order",
                    "drug_order",
                    "annotation_n",
                    "meta_sp_annotation_n",
                    "annotation_mrc_n",
                    "meta_sp_annotation_mrc_n",
                    "core_02",
                    "bed",
                    "vte_status",
                    "hosp_site",
                    "core_resus",
                    "news",
                    "date_time_stamp",
                    "appointments",
                ]
            },
        }
        pert_cols, _ = get_pertubation_columns(
            self.all_df_columns, local_param_dict, self.drop_term_list
        )
        self.assertEqual(len(pert_cols), 0)

    def test_drop_list_population(self):
        """Test that the initial drop_list is populated correctly."""
        local_param_dict = {"outcome_var_n": 1, "data": {}}
        _, drop_list = get_pertubation_columns(
            self.all_df_columns, local_param_dict, self.drop_term_list
        )
        # Should contain '__index_level' and 'Unnamed:' columns
        self.assertIn("some_col__index_level_0", drop_list)
        self.assertIn("Unnamed: 0", drop_list)
        self.assertEqual(len(drop_list), 2)


if __name__ == "__main__":
    unittest.main()
