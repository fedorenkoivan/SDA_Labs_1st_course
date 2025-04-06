#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING_LENGTH 26

typedef struct Node {
    char data[MAX_STRING_LENGTH];
    struct Node* next;
} Node;

Node* create_node(const char* str) {
    Node* new_node = (Node*)malloc(sizeof(Node));
    if (new_node == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }
    strncpy(new_node->data, str, MAX_STRING_LENGTH - 1);
    new_node->data[MAX_STRING_LENGTH - 1] = '\0';
    new_node->next = NULL;
    return new_node;
}

int append(Node** head, const char* str) {
    Node* new_node = create_node(str);
    if (new_node == NULL) {
        return 0;
    }
    
    if (*head == NULL) {
        *head = new_node;
    } else {
        Node* temp = *head;
        while (temp->next) {
            temp = temp->next;
        }
        temp->next = new_node;
    }
    return 1;
}

void print_list(Node* head) {
    Node* temp = head;
    while (temp != NULL) {
        printf("%s ", temp->data);
        temp = temp->next;
    }
    printf("\n");
}

void free_list(Node* head) {
    Node* temp;
    while (head != NULL) {
        temp = head;
        head = head->next;
        free(temp);
    }
}

Node* rearrange(Node* head, int n) {
    if (head == NULL || n <= 0 || n % 20 != 0) {
        return head;
    }

    Node* new_head = NULL;
    Node* new_tail = NULL;
    Node* current = head;

    for (int i = 0; i < n / 20; i++) {
        Node* block_heads[20] = {NULL};

        for (int j = 0; j < 20 && current != NULL; j++) {
            Node* temp = current;
            current = current->next;
            temp->next = NULL;
            block_heads[j] = temp;
        }

        for (int j = 0; j < 10; j++) {
            if (block_heads[j] != NULL) {
                if (new_head == NULL) {
                    new_head = new_tail = block_heads[j];
                } else {
                    new_tail->next = block_heads[j];
                    new_tail = block_heads[j];
                }
            }

            if (block_heads[j + 10] != NULL) {
                if (new_head == NULL) {
                    new_head = new_tail = block_heads[j + 10];
                } else {
                    new_tail->next = block_heads[j + 10];
                    new_tail = block_heads[j + 10];
                }
            }
        }
    }

    return new_head;
}

int main() {
    int n;
    printf("Enter the number of elements (multiple of 20): ");
    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "Failed to read input number\n");
        return 1;
    }

    if (n <= 0 || n % 20 != 0) {
        fprintf(stderr, "Invalid input. The number of elements should be a positive multiple of 20.\n");
        return 1;
    }

    Node* list = NULL;
    char str[MAX_STRING_LENGTH];

    printf("Enter the elements (strings up to %d characters each):\n", MAX_STRING_LENGTH - 1);
    for (int i = 0; i < n; i++) {
        if (scanf("%s", str) != 1) {
            fprintf(stderr, "Failed to read string input\n");
            free_list(list);
            return 1;
        }
        if (!append(&list, str)) {
            fprintf(stderr, "Failed to append node\n");
            free_list(list);
            return 1;
        }
    }

    printf("\nOriginal list:\n");
    print_list(list);

    list = rearrange(list, n);

    printf("\nRearranged list:\n");
    print_list(list);

    free_list(list);
    return 0;
}
