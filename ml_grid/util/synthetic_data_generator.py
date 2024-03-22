import numpy as np
import pandas as pd

columns = ["client_idcode", "timestamp"] + [
    "Insertion - action (qualifier value)_count_subject_present",
    "Alkaline Phosphatase_most-recent",
    "Routine (qualifier value)_count_subject_present",
    "General treatment (procedure)_count_subject_not_present",
    "(2020, 11)_date_time_stamp",
    "(2020, 12)_date_time_stamp",
    "Research fellow (occupation)_count_relative_not_present",
    "RBC_earliest-test",
    "Phlebotomy (procedure)_count_relative_not_present",
    "Date of birth (observable entity)_count_subject_present",
    "Able with difficulty (qualifier value)_count_subject_present",
    "Recurrence (qualifier value)_count",
    "Antibiotic (product)_count",
    "Hypercholesterolemia (disorder)_count",
    "Normal (qualifier value)_count_subject_present_mrc_cs",
    "(1996, 8)_date_time_stamp",
    "Follow-up in outpatient clinic (finding)_count_relative_not_present",
    "Local (qualifier value)_count_relative_not_present",
    "Magnetic resonance imaging of abdomen (procedure)_count_relative_not_present",
    "Identified (qualifier value)_count_relative_not_present",
    "Bisoprolol (substance)_count_subject_not_present",
    "(2021, 9)_date_time_stamp",
    "Reactive (qualifier value)_count_subject_not_present",
    "Left (qualifier value)_count",
    "Physician (occupation)_count_subject_present",
    "Platelet mean volume determination (procedure)_count_relative_not_present",
    "Fracture (morphologic abnormality)_count_relative_not_present",
    "Advanced (qualifier value)_count",
    "Non-smoker (finding)_count",
    "Screening (qualifier value)_count_subject_not_present",
    "Posterior (qualifier value)_count_subject_present",
    "Finding of platelet count (finding)_count_subject_present",
    "Inflammation (qualifier value)_count_relative_not_present",
    "4.4 (qualifier value)_count_subject_present",
    "Medical secretary (occupation)_count_subject_present",
    "Glycated Hb_num-diagnostic-order",
    "Low density lipoprotein cholesterol measurement (procedure)_count_subject_present",
    "Sinus rhythm (finding)_count",
    "Capsule (basic dose form)_count",
    "Massive (qualifier value)_count_relative_not_present",
    "Twice a day (qualifier value)_count_subject_present",
    "Hematology procedure (procedure)_count_subject_not_present",
    "Finding of alkaline phosphatase level (finding)_count_subject_present",
    "Transplantation of liver (procedure)_count",
    "Less-than symbol < (qualifier value)_count_subject_present",
    "(2020, 8)_date_time_stamp",
    "10*12/liter (qualifier value)_count_relative_not_present",
    "Neutrophils_earliest-test",
    "State (environment)_count",
    "Potassium_std",
    "(2013, 1)_date_time_stamp",
    "House (environment)_count",
    "Action (attribute)_count_relative_not_present",
    "Major (qualifier value)_count",
    "(1998, 10)_date_time_stamp",
    "Family (social concept)_count_relative_not_present",
    "C-reactive Protein_min",
    "PLT_days-since-last-test",
    "Occasional (qualifier value)_count_subject_not_present",
    "Foot structure (body structure)_count_subject_not_present",
    "Forest (environment)_count_relative_not_present",
    "Double (qualifier value)_count_relative_not_present",
    "Location (attribute)_count_relative_not_present",
    "Folic acid (substance)_count_relative_not_present",
    "Patient concerned (contextual qualifier) (qualifier value)_count_subject_present_mrc_cs",
    "Past (record artifact)_count",
    "Postoperative period (qualifier value)_count_relative_not_present",
    "Carrier of hemochromatosis (finding)_count_relative_not_present_mrc_cs",
    "White Cell Count_mean",
    "Monocytes_days-between-first-last",
    "Biochemistry ( Bone Profile)_num-diagnostic-order",
    "Hospital admission (procedure)_count_subject_present",
    "Palpitations (finding)_count",
    "Soft (qualifier value)_count_relative_not_present",
    "Less-than symbol < (qualifier value)_count_subject_not_present",
    "Specialist registrar (occupation)_count_subject_present",
    "Evaluation procedure (procedure)_count_relative_not_present",
    "Finding of gamma-glutamyl transferase level (finding)_count_subject_present",
    "Sodium_contains-extreme-high",
    "Activity of daily living (observable entity)_count_subject_present",
    "Liver function tests (observable entity)_count_subject_not_present",
    "Patient (person)_count",
    "Republic of Ireland (geographic location)_count",
    "Dermatologist (occupation)_count_subject_not_present",
    "2.5 (qualifier value)_count",
    "Secretary (occupation)_count_subject_not_present",
    "Father (person)_count",
    "week (qualifier value)_count_subject_not_present",
    "(2007, 9)_date_time_stamp",
    "History of clinical finding in subject (situation)_count_relative_not_present",
    "PLT_median",
    "Information (qualifier value)_count_subject_not_present",
    "(2017, 5)_date_time_stamp",
    "MCHC._mode",
    "Calcium measurement (procedure)_count",
    "(2018, 4)_date_time_stamp",
    "Request (record artifact)_count_relative_not_present",
    "Principal diagnosis (contextual qualifier) (qualifier value)_count",
    "Hospital department (environment)_count_subject_present",
    "outcome_var_1",
]


def generate_time_series(num_clients, num_rows_per_client):
    # Generate client IDs
    client_ids = list(range(1, num_clients + 1))

    # Generate dates
    date_range = pd.date_range(
        start="2022-01-01", periods=num_rows_per_client, freq="D"
    )

    # Create an empty list to store client data
    client_data_list = []

    # Generate data for each client
    for client_id in client_ids:
        client_data = []
        for i in range(num_rows_per_client):
            # Generate random noise for features
            features = np.random.uniform(-1, 1, 99)
            # Generate target variable
            target = np.random.randint(0, 2)  # Binary target variable
            # Add association between features and target
            if target == 1:
                # Increase feature values for positive targets
                features += 1.5
            client_data.append([client_id, date_range[i]] + list(features) + [target])
        # Append client data to the list
        client_data_list.extend(client_data)

    # Create DataFrame from the list of client data
    df = pd.DataFrame(client_data_list, columns=columns)

    # Sort the DataFrame by timestamp within each client_id group
    df_sorted = df.sort_values(by=["client_idcode", "timestamp"]).reset_index(drop=True)

    return df_sorted


# Generate example data
# example_data = generate_time_series(num_clients=10, num_rows_per_client=5)
# example_data.head()

# example_data.to_csv('unit_test_synthetic_time_series_data.csv', index=False)
