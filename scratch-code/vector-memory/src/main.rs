use std::mem;

fn main() {
    let vec = vec![10, 20, 30, 40];

    // Call the function to print elements and their pointers, and to check for padding
    print_elements_and_check_padding(&vec);
}

fn print_elements_and_check_padding(vec: &[i32]) {
    // Check if the vector is empty or has only one element
    if vec.len() < 2 {
        println!("Vector needs at least two elements to check for padding.");
        return;
    }

    // Obtain the base pointer of the vector
    let base_ptr = vec.as_ptr();
    
    // Get the size of an element
    let elem_size = mem::size_of::<i32>();

    // Iterate over the vector elements
    for i in 0..vec.len() - 1 {
        let ptr1 = unsafe { base_ptr.add(i) };
        let ptr2 = unsafe { base_ptr.add(i + 1) };

        // Calculate the difference in bytes between the two pointers
        let diff = (ptr2 as usize) - (ptr1 as usize);

        println!("Element {}: Value = {}, Pointer = {:?}, Difference to next = {} bytes, Size of element = {} bytes", 
                 i, vec[i], ptr1, diff, elem_size);
    }

    // Print the last element without a subsequent element
    let last_ptr = unsafe { base_ptr.add(vec.len() - 1) };
    println!("Element {}: Value = {}, Pointer = {:?}, Size of element = {} bytes", vec.len() - 1, vec[vec.len() - 1], last_ptr, elem_size);
}
