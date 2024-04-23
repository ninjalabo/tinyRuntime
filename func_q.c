#include <math.h>

#include "properties.h"
#include "func_common.h"

static float vec_dot(int8_t *q, float *sq, int8_t *w, float *sw, int size,
		     int gs) 
{
        int ngroups = size / gs;
        float val = 0.0f;
        for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		int32_t ival = 0;
		for (int k = 0; k < gs; k++) {
			ival +=
			    ((int32_t) q[start_idx + k]) *
			    ((int32_t) w[start_idx + k]);
		}
		val += ((float) ival) * sq[group] * sw[group];
        }

	return val;
}

void linear_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
	      LinearConfigQ lc)
{
	// w(nrows,ncols) @ x (ncols,) + b(nrows,) -> xout (nrows,)
	int ncols = lc.in;
	int nrows = lc.out;
	int gs_w = lc.gs_weight;
	int gs_b = lc.gs_bias;

	int8_t *w = p + lc.qoffset;
	int8_t *b = w + ncols * nrows;
	float *sw = sf + lc.soffset;
	float *sb = sw + ncols * nrows / gs_w;
	for (int bs = 0; bs < batch_size; bs++) {
		for (int i = 0; i < nrows; i++) {
			float val =
			    vec_dot(&x->q[bs * ncols], &x->s[bs * ncols / gs_w],
				    &w[i * ncols], &sw[i * ncols / gs_w], ncols,
				    gs_w);
			float bias_val =
			    gs_b > 0 ? ((float) b[i]) * sb[i / gs_b] : 0.0f;
			xout[bs * nrows + i] = val + bias_val;
		}
	}
}

void conv_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
	    ConvConfigQ cc, int height, int width)
{
	// w (nchannels,in) @ x (nrows,ncols) + b(nchannels,) -> xout (nchannels,ncols) <-> xout (nchannels, height, width)
	int nchannels = cc.oc;
	int ncols = height * width;
	int nrows = cc.ic * cc.ksize * cc.ksize;
	int gs_w = cc.gs_weight;
	int gs_b = cc.gs_bias;

	int8_t *w = p + cc.qoffset;
	int8_t *b = w + nchannels * nrows;
	float *sw = sf + cc.soffset;
	float *sb = sw + nchannels * nrows / gs_w;
	for (int bs = 0; bs < batch_size; bs++) {
		int x_idx = bs * nrows * ncols;
		int xout_idx = bs * nchannels * ncols;
		for (int c = 0; c < nchannels; c++) {
			for (int i = 0; i < ncols; i++) {
				float val =
				    vec_dot(&x->q[x_idx + i * nrows],
					    &x->s[(x_idx + i * nrows) / gs_w],
					    &w[c * nrows],
					    &sw[c * nrows / gs_w], nrows, gs_w);
				float bias_val =
				    gs_b > 0 ? ((float) b[c]) * sb[i / gs_b] :
				    0.0f;
				xout[xout_idx + c * ncols + i] = val + bias_val;
			}
		}
	}
}

static float quantize_find_wmax(float *x, int gs, int start_idx)
{
	// find the maximum absolute value in the current group
	float wmax = 0.0;
	for (int i = 0; i < gs; i++) {
		float val = fabsf(x[i + start_idx]);
		wmax = wmax > val ? wmax : val;
	}
	return wmax;
}

static void quantize_scale(QuantizedTensor * qx, float *x, int gs,
			   int start_idx, float scale)
{
	// scale and save weights
	for (int i = 0; i < gs; i++) {
		float quant_value = x[i + start_idx] / scale;
		qx->q[i + start_idx] = roundf(quant_value);
	}
}

void quantize(QuantizedTensor * qx, float *x, int n, int gs)
{
	int ngroups = batch_size * n / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		float wmax = quantize_find_wmax(x, gs, start_idx);
		float scale = wmax / Q_MAX;
		qx->s[group] = scale;
		quantize_scale(qx, x, gs, start_idx, scale);
	}
}

void quantize2d(QuantizedTensor * qx, float *x, ConvConfigQ cc, int nrows)
{
	int ncols = cc.ic * cc.ksize * cc.ksize;
	int gs = cc.gs_weight;
	int ngroups = batch_size * nrows * ncols / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < ngroups; group++) {
		int start_idx = group * gs;
		float wmax = quantize_find_wmax(x, gs, start_idx);
		float scale = wmax / Q_MAX;
		qx->s[group] = scale;
		quantize_scale(qx, x, gs, start_idx, scale);
	}
}
