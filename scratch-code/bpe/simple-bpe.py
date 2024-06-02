"""
 Example:
    Suppose the data to be encoded is
    ```text
    aaabdaaabac
    ```
    The byte pair "aa" occurs most often, so it will be replaced by a byte that is not used in the data, such as "Z". Now there is the following data and replacement table:
    ```text
    ZabdZabac
    Z=aa
    ```
    Then the process is repeated with byte pair "ab", replacing it with "Y":
    ```
    ZYdZYac
    Y=ab
    Z=aa
    ```
    The only literal byte pair left occurs only once, and the encoding might stop here. Alternatively, the process could continue with recursive byte pair encoding, replacing "ZY" with "X":
    ```text
    XdXac
    X=ZY
    Y=ab
    Z=aa
    ```
    This data cannot be compressed further by byte pair encoding because there are no pairs of bytes that occur more than once.
    To decompress the data, simply perform the replacements in the reverse order.
"""

from collections import defaultdict


def most_frequent_pair_once(input_str: str, replacement_str: str): 
    total_str = len(input_str)
    lookup = defaultdict(int)

    for pair in zip(input_str[0:total_str - 1], input_str[1:total_str]):
        key = ''.join(pair)
        lookup[key] += 1

    max_key = max(lookup, key=lookup.get)
    max_value = lookup[max_key]

    if max_value > 1:
        return input_str.replace(max_key, replacement_str)
    else:
        return input_str

if __name__ == "__main__":
    input_str_v0 = "aaabdaaabac"
    input_str_v1 = most_frequent_pair_once(input_str_v0, "Z")
    input_str_v2 = most_frequent_pair_once(input_str_v1, "Y")
    input_str_v3 = most_frequent_pair_once(input_str_v2, "X")
    input_str_v4 = most_frequent_pair_once(input_str_v3, "A")

    print("input_str_v0 = ", input_str_v0)
    print("input_str_v1 = ", input_str_v1)
    print("input_str_v2 = ", input_str_v2)
    print("input_str_v3 = ", input_str_v3)
    print("input_str_v4 = ", input_str_v4)

    


