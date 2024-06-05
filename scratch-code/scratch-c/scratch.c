#include <stdio.h>


int main() {

    unsigned int ui = 10;
    unsigned long ul = 10;
    unsigned long long ull = 10;

    printf("sizeof(unsigned int) = %d\n", (int) sizeof(unsigned int) * 8); 
    printf("sizeof(unsigned long) = %d\n", (int) sizeof(unsigned long) * 8); 
    printf("sizeof(unsigned long long) = %d\n", (int) sizeof(unsigned long long) * 8);

    return 0;
}
