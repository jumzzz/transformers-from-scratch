#include <stdio.h>
#include <stdint.h>

// Function to print byte representation of a value
void print_bytes(const void *value, size_t size) {
    const uint8_t *byte_pointer = (const uint8_t *)value;
    for (size_t i = 0; i < size; ++i) {
        printf("%02X ", byte_pointer[i]);
    }
    printf("\n");
}

int main() {
    print_bytes((int*)0xC0, 1);
    return 0;
}
