#include <stdio.h>
#include <math.h>

struct params_data;

struct params_data {
    int n;
    double *params;
};

double f(int n, double *x, void *user_data) {
    struct params_data *pd = (struct params_data *)user_data;
    double K = pd->params[0];
    double cum = 0;
    for (int i = 1; i < pd->n; i++) {
        cum = cum + pd->params[i] * pow(x[0], i);
    }
    return K * exp(cum);
}