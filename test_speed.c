#include "func.h"
#include "func_q.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_BUF_SIZE (64 * 3 * 3 * 28 * 28) // size needed in im2col and conv
#define MAX_PARAM_SIZE (512 * 512 * 512) // parameter size in linear

static float x[MAX_BUF_SIZE];
static float x2[MAX_BUF_SIZE];
static int8_t xq_q[MAX_BUF_SIZE];
static float xq_s[MAX_BUF_SIZE];
static float p[MAX_PARAM_SIZE];
static int8_t q[MAX_PARAM_SIZE];

        
int main(int argc, char **argv)
{
        // command: ./pt <function> <niter>
        if (argc < 2) {
		fprintf(stderr, "Give function name as an argument\n");
	        exit(EXIT_FAILURE);
	}
	char *function = argv[1];

        int niter = 30;
        if (argc == 3) {
                niter = atoi(argv[2]);
        }

        // initialize buffers and parameters
        for (int i = 0; i < MAX_BUF_SIZE; i++) {
                float fval = (float) (rand() / RAND_MAX);
                x[i] = fval;
                x2[i] = fval;
                xq_s[i] = fval;
                xq_q[i] = rand() % 255 - 127;
        }
        QuantizedTensor xq = {
             .q = xq_q,
             .s = xq_s
        };
        for (int i = 0; i < MAX_PARAM_SIZE; i++) {
                p[i] = (float) (rand() / RAND_MAX);
                q[i] = rand() % 255 - 127;
        }

        // record time taken by a function
        clock_t start, end;
        start = clock();

        for (int i = 0; i < niter; i++) {
                if (strcmp(function, "linear") == 0) {
                        int dim = 512;
                        LinearConfig lc = { dim, dim, 0, 1 };
                        linear(x2, x, p, lc);
                }
                if (strcmp(function, "linear_q") == 0) {
                        int dim = 512, gs_w = 64, gs_b = 64;
                        LinearConfigQ lc = { dim, dim, 0, 0, gs_w, gs_b };
                        linear_q(x, &xq, q, p, lc);
                }
                if (strcmp(function, "conv") == 0) {
                        int ksize = 3, stride = 1, pad = 1, ic = 64, oc = 64;
                        int h = 28, w = 28;
                        ConvConfig cc = { ksize, stride, pad, ic, oc, 0, 1 };
                        conv(x2, x, p, cc, h, w);
                }
                if (strcmp(function, "conv_q") == 0) {
                        int ksize = 3, stride = 1, pad = 1, ic = 64, oc = 64;
                        int gs_w = 64, gs_b = 64, h = 28, w = 28;
                        ConvConfigQ cc =
                        { ksize, stride, pad, ic, oc, 0, 0, gs_w, gs_b };
                        conv_q(x, &xq, q, p, cc, h, w);
                }
                if (strcmp(function, "matadd") == 0) {
                        matadd(x2, x, MAX_BUF_SIZE);
                }
                if (strcmp(function, "im2col") == 0) {
                        int ksize = 3, stride = 1, pad = 1, ic = 64;
                        int h = 28, w = 28;
                        ConvConfig cc = { ksize, stride, pad, ic, 0, 0, 0 };
                        im2col(x2, x, cc, &h, &w);
                }
                if (strcmp(function, "im2col_q") == 0) {
                        int ksize = 3, stride = 1, pad = 1, ic = 64;
                        int h = 28, w = 28;
                        ConvConfigQ cc =
                            { ksize, stride, pad, ic, 0, 0, 0, 0, 0 };
                        im2col_q(x2, x, cc, &h, &w);
                }
                if (strcmp(function, "batchnorm") == 0) {
                        int nch = 64, h = 56, w = 56;
                        BnConfig bc = { nch, 0 };
                        batchnorm(x2, x, p, bc, h, w);
                }
                if (strcmp(function, "maxpool") == 0) {
                        int nch = 64, h = 56, w = 56;
                        int ksize = 3, stride = 1, pad = 1;
                        maxpool(x2, x, &h, &w, nch, ksize, stride, pad);
                }
                if (strcmp(function, "avgpool") == 0) {
                        int nch = 64, h = 56, w = 56;
                        int ksize = 3, stride = 1, pad = 1;
                        avgpool(x2, x, &h, &w, nch, ksize, stride, pad);
                }
                if (strcmp(function, "relu") == 0) {
                        relu(x, MAX_BUF_SIZE);
                }
                if (strcmp(function, "softmax") == 0) {
                        softmax(x, MAX_BUF_SIZE);
                }
                if (strcmp(function, "quantize") == 0) {
                        int gs = 64;
                        quantize(&xq, x, MAX_BUF_SIZE, gs);
                }
                if (strcmp(function, "quantize2d") == 0) {
                        int ksize = 3, ic = 64, gs = 64, h = 28, w = 28;
                        ConvConfigQ cc = { ksize, 0, 0, ic, 0, 0, 0, gs, 0};
                        quantize2d(&xq, x, cc, h * w);
                }
        }

        end = clock();
        float cpu_time = (float) (end - start) / CLOCKS_PER_SEC;
        printf("CPU time used: %f seconds\n", cpu_time);

	return 0;
}
