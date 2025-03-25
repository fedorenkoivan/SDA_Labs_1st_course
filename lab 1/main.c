#include <stdio.h>
#include <math.h>

double sqrt_series_descent(double x, int n, int i, double term, double sum) {
    sum += term;
    printf("f_%d = %.10lf, sum = %.10lf\n", i, term, sum);

    if (i == n - 1) {
        return sum;
    }

    double next_term = term * ((3.0 - 2.0 * (i + 1)) / (2.0 * (i + 1))) * (x - 1);
    return sqrt_series_descent(x, n, i + 1, next_term, sum);
}

typedef struct {
    double term;
    double sum;
} Result;

Result sqrt_series_return(double x, int n, int i) {
    if (i == 0) {
        printf("f_0 = %.10lf\n", 1.0);
        return (Result){1.0, 1.0};
    }

    Result prev = sqrt_series_return(x, n, i - 1);
    double t = x - 1;
    double term = prev.term * ((3.0 - 2.0 * i) / (2.0 * i)) * t;
    double sum = prev.sum + term;

    printf("f_%d = %.10lf, sum = %.10lf\n", i, term, sum);
    return (Result){term, sum};
}

double sqrt_series_hybrid(double x, int n, int i, double prev_term) {
    double term = (i == 0) ? 1.0 :
        prev_term * ((3.0 - 2.0 * i) / (2.0 * i)) * (x - 1);

    if (i == n - 1) {
        printf("f_%d = %.10lf, sum = %.10lf\n", i, term, term);
        return term;
    }

    double rest_sum = sqrt_series_hybrid(x, n, i + 1, term);
    double total = term + rest_sum;

    printf("f_%d = %.10lf, sum = %.10lf\n", i, term, total);
    return total;
}

double sqrt_series_loop(int n, double x) {
    double sum = 1.0; // f_0 = 1
    double term = 1.0;

    for (int i = 1; i < n; i++) {
        term *= ((3.0 - 2.0 * i) / (2.0 * i)) * (x - 1);
        sum += term;
        printf("f_%d = %.10lf, sum = %.10lf\n", i, term, sum);
    }

    return sum;
}

int main() {
    int n = 5;
    double x = 0.79;
    if(x > 0.5 && x < 1) {
        printf("\nN = %d, X = %.3lf.", n, x);
        printf("\n--- [1] Method1 ---\n");
        double res1 = sqrt_series_descent(x, n, 0, 1.0, 0.0);
        printf("Final sum = %.10lf\n", res1);
        
        printf("\n--- [2] Method2 ---\n");
        Result res2 = sqrt_series_return(x, n, n - 1);
        printf("Final sum = %.10lf\n", res2.sum);

        printf("\n--- [3] Method3 ---\n");
        double res3 = sqrt_series_hybrid(x, n, 0, 0.0);
        printf("Final sum = %.10lf\n", res3);

        printf("\n--- [4] Method4 ---\n");
        double res4 = sqrt_series_loop(n, x);
        printf("Final sum = %.10lf\n", res4);

        double actual = sqrt(x);
        printf("\nActual sqrt(%.2lf) = %.10lf\n", x, actual);
        printf("\nError: %.10lf", fabs(res1 - actual));
    } else {
        printf("\nmIvalid data: %d, %.3lf", n, x);
    }

    return 0;
}