#include <stdio.h>

int main() {
    int i = 0;
    int j = 0;
    
    while (i < 10) printf("%d ", i++);
    printf("\n");
    while (j < 10) printf("%d ", ++j);
    printf("\n");

    return 0;
}
