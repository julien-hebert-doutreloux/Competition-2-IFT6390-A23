import numpy as np
import pandas as pd

def transform_labels(label_A, label_B):
    # Compute the ASCII sum of the labels and add them together
    ascii_sum = ord('A') + label_A + ord('A') + label_B

    # Adjust the sum if it exceeds the range 122 (ASCII for 'z')
    while ascii_sum > 122:
        ascii_sum -= 65
    # Convert the ASCII sum to its corresponding character
    final_char = chr(ascii_sum)

    # Convert the character to string dtype
    return str(final_char)
