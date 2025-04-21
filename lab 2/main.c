#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING_LENGTH 26
#define BLOCK 20
#define HALF  (BLOCK/2)

typedef struct Node {
    char data[MAX_STRING_LENGTH];
    struct Node* next;
} Node;

Node* create_node(const char* str) {
    Node* new_node = malloc(sizeof(Node));
    if (!new_node) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    strncpy(new_node->data, str, MAX_STRING_LENGTH - 1);
    new_node->data[MAX_STRING_LENGTH - 1] = '\0';
    new_node->next = NULL;
    return new_node;
}

void append(Node** head, const char* str) {
    Node* nd = create_node(str);
    if (*head == NULL) {
        *head = nd;
    } else {
        Node* t = *head;
        while (t->next) t = t->next;
        t->next = nd;
    }
}

void print_list(Node* head) {
    for (; head; head = head->next) {
        printf("%s ", head->data);
    }
    printf("\n");
}

void free_list(Node* head) {
    while (head) {
        Node* t = head;
        head = head->next;
        free(t);
    }
}

Node* rearrange(Node* head) {
    Node dummy;
    Node* tail = &dummy;
    dummy.next = NULL;

    Node* rest = head;

    while (rest) {
        Node* first_half = rest;
        Node* mid = rest;
        
        for (int i = 0; i < HALF; i++) {
            if (!mid) return dummy.next;
            mid = mid->next;
        }
        
        Node* block_tail = mid;
        for (int i = 0; i < HALF - 1; i++) {
            if (!block_tail) return dummy.next;
            block_tail = block_tail->next;
        }
        Node* next_block = block_tail ? block_tail->next : NULL;
        if (block_tail) block_tail->next = NULL;
        
        while (first_half && mid) {
            Node* first = first_half;
            first_half = first_half->next;
            first->next = NULL;
            tail->next = first;
            tail = first;
            
            Node* second = mid;
            mid = mid->next;
            second->next = NULL;
            tail->next = second;
            tail = second;
        }
        
        rest = next_block;
    }

    return dummy.next;
}

int main(void) {
    int n;
    printf("Enter number of elements (positive multiple of %d): ", BLOCK);
    if (scanf("%d", &n) != 1 || n <= 0 || n % BLOCK != 0) {
        fprintf(stderr, "Bad input: n must be positive multiple of %d\n", BLOCK);
        return EXIT_FAILURE;
    }

    Node* list = NULL;
    char buf[MAX_STRING_LENGTH];

    printf("Enter %d strings (up to %d chars each):\n", n, MAX_STRING_LENGTH - 1);
    for (int i = 0; i < n; i++) {
        if (scanf("%25s", buf) != 1) {
            fprintf(stderr, "Failed to read string\n");
            free_list(list);
            return EXIT_FAILURE;
        }
        append(&list, buf);
    }

    printf("\nOriginal list:\n");
    print_list(list);

    Node* new_list = rearrange(list);

    printf("\nRearranged list:\n");
    print_list(new_list);

    free_list(new_list);
    return EXIT_SUCCESS;
}