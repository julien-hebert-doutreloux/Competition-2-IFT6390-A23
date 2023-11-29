import numpy as np
import pandas as pd

def transform_labels(label_A, label_B):
    # Compute the ASCII sum of the labels and add them together
    capital_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    ascii_sum = ord(capital_alphabet[label_A]) + ord(capital_alphabet[label_B])

    # Adjust the sum if it exceeds the range 122 (ASCII for 'z')
    while ascii_sum > 122:
        ascii_sum -= 65
    # Convert the ASCII sum to its corresponding character
    final_char = chr(ascii_sum)

    # Convert the character to string dtype
    return str(final_char)
