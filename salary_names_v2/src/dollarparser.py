import re
import numpy as np

import re
import numpy as np

def parse_dollar_amount(x) -> int:
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
        if average_amount > 10 and average_amount < 150:
            return int(round(average_amount))*1000
        elif np.isinf(average_amount) or np.isnan(average_amount):
            return np.nan 
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
