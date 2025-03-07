/*
* This file is part of the LSTM Network implementation In C made by Rickard Hallerbäck
* 
*                 Copyright (c) 2018 Rickard Hallerbäck
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), 
* to deal in the Software without restriction, including without limitation the rights 
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
* Software, and to permit persons to whom the Software is furnished to do so, subject to 
* the following conditions:
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
*
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
* OR OTHER DEALINGS IN THE SOFTWARE.
*/

/* 
* Dealing with FC layers, forward and backward
*
* ==================== NOTE: ====================
*   The caller should have thought about the memory 
*    allocation, these functions assumes that 
*           everything is OK. 
* =================================================
*
*/

#include "layers.h"
#include <omp.h>

#ifdef WINDOWS
#include <stdio.h>
#endif

//    Y = AX + b        &Y,      A,       X,    B,     Rows (for A), Columns (for A)

void fully_connected_forward(double* Y, double* A, double* X, double* b, int R, int C) {
    #pragma omp parallel for
    for (int i = 0; i < R; i++) {
        double sum = b[i];  // Initialize with bias
        for (int n = 0; n < C; n++) {
            sum += A[i * C + n] * X[n];
        }
        Y[i] = sum;  // Store the result
    }
}

//    Y = AX + b        dldY,       A,     X,        &dldA,    &dldX,    &dldb   Rows (A), Columns (A)

void fully_connected_backward(double* dldY, double* A, double* X, double* dldA,
                              double* dldX, double* dldb, int R, int C) {
    // Computing dldA
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < R; i++) {
        for (int n = 0; n < C; n++) {
            dldA[i * C + n] = dldY[i] * X[n];
        }
    }

    // Computing dldb
    #pragma omp parallel for
    for (int i = 0; i < R; i++) {
        dldb[i] = dldY[i];
    }

    // Computing dldX
    #pragma omp parallel for
    for (int i = 0; i < C; i++) {
        double sum = 0.0;
        for (int n = 0; n < R; n++) {
            sum += A[n * C + i] * dldY[n];
        }
        dldX[i] = sum;
    }
}


double cross_entropy(double* probabilities, int correct)
{
  return -log(probabilities[correct]);  
}

// Dealing with softmax layer, forward and backward
//                &P,   Y,    features


void softmax_layers_forward(double* P, double* Y, int F, double temperature) {
    double sum = 0.0;

#ifdef WINDOWS
    double *cache = malloc(sizeof(double) * F);
    if (cache == NULL) {
        fprintf(stderr, "%s.%s.%d malloc(%zu) failed\r\n", 
                __FILE__, __func__, __LINE__, sizeof(double) * F);
        exit(1);
    }
#else
    double cache[F]; // Stack allocation for non-Windows compilers
#endif

    // Compute exponentials in parallel
    #pragma omp parallel for reduction(+:sum)
    for (int f = 0; f < F; f++) {
        cache[f] = exp(Y[f] / temperature);
        sum += cache[f];
    }

    // Normalize in parallel
    #pragma omp parallel for
    for (int f = 0; f < F; f++) {
        P[f] = cache[f] / sum;
    }

#ifdef WINDOWS
    free(cache);
#endif
}

//                    P,    c,  &dldh, rows
void  softmax_loss_layer_backward(double* P, int c, double* dldh, int R)
{ 
  int r = 0;

  while ( r < R ) {
    dldh[r] = P[r];
    ++r;
  }

  dldh[c] -= 1.0;
}
// Other layers used: sigmoid and tanh
//  
//    Y = sigmoid(X), &Y, X, length

void sigmoid_forward(double* Y, double* X, int L) {
    #pragma omp parallel for
    for (int l = 0; l < L; l++) {
        Y[l] = 1.0 / (1.0 + exp(-X[l]));
    }
}

void sigmoid_backward(double* dldY, double* Y, double* dldX, int L) {
    #pragma omp parallel for
    for (int l = 0; l < L; l++) {
        dldX[l] = (1.0 - Y[l]) * Y[l] * dldY[l];
    }
}

void tanh_forward(double* Y, double* X, int L) {
    #pragma omp parallel for
    for (int l = 0; l < L; l++) {
        Y[l] = tanh(X[l]);
    }
}

void tanh_backward(double* dldY, double* Y, double* dldX, int L) {
    #pragma omp parallel for
    for (int l = 0; l < L; l++) {
        dldX[l] = (1.0 - Y[l] * Y[l]) * dldY[l];
    }
}

