/**
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
 * ***/

use std::collections::HashMap;


fn replace_first_occurrence(s: &str, from: &str, to: &str) -> String {
    if let Some(pos) = s.find(from) {
        let mut result = String::with_capacity(s.len() - from.len() + to.len());
        result.push_str(&s[..pos]);
        result.push_str(to);
        result.push_str(&s[pos + from.len()..]);
        result
    } else {
        s.to_string()
    }
}


fn compress_to_pair_once(input_str: &String) -> HashMap<String, usize> {
    let total_len = input_str.len();
    let mut lookup: HashMap<String, usize> = HashMap::new();

    for i in 0..total_len - 1 {
        let slice = &input_str[i..i+2];
        let key = slice.to_string();
        *lookup.entry(key).or_insert(1) += 1;
    }



    return lookup
}


fn main() {
    println!("{:?}", compress_to_pair_once(&"aaabdaaabac".to_string()));
}
