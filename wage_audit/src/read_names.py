import pandas as pd
import numpy as np
import logging

def read_names(names_fn):
    """Reads applicant/name data from the Excel file, matching first and last names by race."""
    try:
        # Load first and last names from the Excel file
        df_first = pd.read_excel(names_fn, sheet_name="first name")
        df_last = pd.read_excel(names_fn, sheet_name="last name")
        
        # Group first and last names by race
        first_names_by_race = df_first.groupby('Race')
        last_names_by_race = df_last.groupby('Race')
        
        # Combine first and last names only within the same race
        applicants = []
        for race in first_names_by_race.groups:
            first_names = first_names_by_race.get_group(race)
            last_names = last_names_by_race.get_group(race)
            
            for _, first_row in first_names.iterrows():
                for _, last_row in last_names.iterrows():
                    full_name = f"{first_row['First Name']} {last_row['Last name']}"
                    applicant = {
                        'Full Name': full_name,
                        'Gender': first_row['Gender'],
                        'Race': race  # The race is common between both first and last names
                    }
                    applicants.append(applicant)
        return applicants
    except Exception as e:
        logging.error(f"Error reading names from audit_names.xlsx: {e}")
        sys.exit(1)

