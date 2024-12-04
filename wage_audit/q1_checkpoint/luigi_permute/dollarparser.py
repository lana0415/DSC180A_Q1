import re
import numpy as np

import re
import numpy as np

import numpy as np
import pandas as pd
import re

def parse_dollar_amount(x) -> int:
    """
    Function to extract dollar amounts from a string and return their average.
    Handles ranges like "$70,000 to $80,000" by averaging them. Designed for
    parsing annual salaries, considering only amounts between $30,000 and $500,000.
    Ignores amounts outside this range.

    Parameters:
    x (str): The string containing the dollar amount(s).

    Returns:
    int: The average of the valid dollar amounts, or np.nan if none are valid.
    """
    if pd.isna(x):
        return np.nan

    x = x.lower().strip()
    # Normalize separators and remove irrelevant words
    x = x.replace('–', '-').replace('—', '-').replace('−', '-')
    x = x.replace('to', '-').replace('and', '-')
    x = re.sub(r'per\s+(year|annum|month|week|hour)', '', x)
    x = re.sub(r'[^$\d\.\-,kmbmillionthousand ]', '', x)

    # Regex pattern to match amounts
    pattern = r'\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb]|thousand|million|billion)?'
    matches = re.findall(pattern, x)

    if not matches:
        return np.nan

    amounts = []
    for amount_str, magnitude in matches:
        amount_num = float(amount_str.replace(',', ''))
        magnitude = magnitude.lower() if magnitude else ''
        # Apply magnitudes
        if magnitude in ['k', 'thousand']:
            amount_num *= 1e3
        elif magnitude in ['m', 'million']:
            amount_num *= 1e6
        elif magnitude in ['b', 'billion']:
            amount_num *= 1e9
        amounts.append(amount_num)

    # Filter amounts between $30,000 and $500,000
    valid_amounts = [amount for amount in amounts if 30000 <= amount <= 500000]

    if not valid_amounts:
        return np.nan

    # Handle ranges by averaging the first two valid amounts if a range is indicated
    if '-' in x and len(valid_amounts) >= 2:
        average_amount = sum(valid_amounts[:2]) / 2
    else:
        average_amount = sum(valid_amounts) / len(valid_amounts)

    return int(round(average_amount))


def parse_dollar_amount_old_v1(x) -> int:
    """
    Function to extract dollar amounts from a string and return their average.
    Handles ranges like "$70,000 to $80,000" by averaging them. Ignores qualitative
    descriptors like "mid-", "high-", or "low-", focusing only on the numerical values.
    Designed for parsing annual salaries, typically ranging from $30,000 to $250,000.

    Parameters:
    x (str): The string containing the dollar amount(s).
    
    Returns:
    int: The average of the mentioned dollar amounts, converted to an integer.
    """
    if x is np.nan:
        return np.nan
    try:
        result = x.lower().strip()
    except Exception as e:
        print("ERROR!", x, e)
    
    # Regex to handle numbers, possible ranges, and magnitude identifiers more robustly
    matches = re.finditer(r'\$\s*(\d{1,3}(?:\s*,\s*\d{3})*(?:-\d{1,3}(?:\s*,\s*\d{3})*)?)(\.\d+)?\s*([kmb]|million|thousand)?\b', result)
    amounts = []

    for match in matches:
        amount, fraction, magnitude = match.groups()
        # Handle ranges by averaging them
        if '-' in amount:
            values = [float(val.replace(',', '').replace(' ', '') + (fraction if fraction else '')) for val in amount.split('-')]
            amount = sum(values) / len(values)  # Average the range
        else:
            amount = float(amount.replace(',', '').replace(' ', '') + (fraction if fraction else ''))

        # Apply magnitude
        if magnitude:
            if 'thousand' in magnitude or 'k' in magnitude:
                amount *= 1000
            elif 'million' in magnitude or 'm' in magnitude:
                amount *= 1000000
        
        amounts.append(amount)
        
    

    # Average all found amounts
    if amounts:
        average_amount = sum(amounts) / len(amounts)
        if average_amount > 30 and average_amount < 250:
            return int(round(average_amount))*1000
        return int(round(average_amount))
    else:
        return np.nan




def parse_dollar_amount_new(x) -> int:
    """
    Enhanced function to extract the first dollar amount mentioned
    in a string. Handles formats like "$25k to $35k" by returning
    the first amount. It also interprets shorthand for thousands 
    ('k') and millions ('m' or 'million').

    Parameters:
    x (str): The string containing the dollar amount.
    
    Returns:
    int: The numeric value of the first mentioned dollar amount, converted to an integer.
    """
    if x is np.nan:
        return np.nan
    try:
        result = x.lower().strip()
    except Exception as e:
        print("ERROR!", x, e)
    # Adjust regex to handle spaces inside and around the numbers more effectively
    match = re.search(r'\$\s*(\d{1,3}(?:\s*,\s*\d{3})*)(\.\d+)?\s*([kmb]|million|thousand)?', result)
    if not match:
        return np.nan

    # Clean up the result to remove spaces and other non-numeric characters
    amount, fraction, magnitude = match.groups()
    amount = amount.replace(',', '').replace(' ', '')
    result = amount + (fraction if fraction else '')

    # Handling thousands and millions
    if magnitude:
        if 'thousand' in magnitude or 'k' in magnitude:
            result = float(result) * 1000
        elif 'million' in magnitude or 'm' in magnitude:
            result = float(result) * 1000000
    # Convert to integer
    try:
        return int(round(float(result)))
    except ValueError:
        return np.nan



def parse_dollar_amount_old(x) -> int:
    """
    Enhanced function to extract the first dollar amount mentioned
    in a string. Handles formats like "$25k to $35k" by returning
    the first amount. It also interprets shorthand for thousands 
    ('k') and millions ('m' or 'million').

    Parameters:
    x (str): The string containing the dollar amount.
    
    Returns:
    int: The numeric value of the first mentioned dollar amount, converted to an integer.
    """
    if x is np.nan:
        return np.nan
    try:
        result = x.lower().strip()
    except Exception as e:
        print("ERROR!", x, e)
    # Finding the first occurrence of a dollar amount
    match = re.search(r'\$\d{1,3}(?:,\d{3})*(\.\d+)?([kmb]| million| thousand)?', result)
    if not match:
        #print(f"No dollar amount found in the string (returning np.nan): {x}")
        return np.nan

    result = match.group(0).replace(',', '').replace('$', '')
    
    # Handling thousands and millions
    if 'thousand' in result or 'k' in result:
        result = re.sub('[^\d.]', '', result)
        result = float(result) * 1000
    elif 'million' in result or 'm' in result:
        result = re.sub('[^\d.]', '', result)
        result = float(result) * 1000000
    else:
        result = re.sub('[^\d.]', '', result)  # Remove non-numeric characters


    # Convert to integer
    try:
        return int(round(float(result)))
    except ValueError:
        return np.nan
