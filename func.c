#include <math.h>
#include <stdbool.h>

#include "properties.h"

// avoid division by zero
#define eps 0.00001f

void linear(float *xout, float *x, float *p, LinearConfig lc, bool relu)
{
	// w(out,in) @ x (in,) + b(out,) -> xout (out,)
	int in = lc.in;
	int out = lc.out;

	float *w = p + lc.offset;
	float *b = w + in * out;
	for (int i = 0; i < out; i++) {
		float val = 0.0f;
		for (int j = 0; j < in; j++) {
			val += w[i * in + j] * x[j];
		}
		float bias_val = lc.bias ? b[i] : 0.0f;
		xout[i] = (relu) ? fmax(val + bias_val, 0.0f) : val + bias_val;
	}
}

static float linear_q_matmul(QuantizedTensor * x, int8_t * w, float *sw,
			     int in, int gs_w, int i)
{
	float val = 0.0f;
	int32_t ival = 0;
	// do the matmul in groups of gs_w
	for (int j = 0; j < in; j += gs_w) {
		for (int k = 0; k < gs_w; k++) {
			ival +=
			    ((int32_t) x->q[j + k]) *
			    ((int32_t) w[i * in + j + k]);
		}
		val += ((float)ival) * sw[(i * in + j) / gs_w] * x->s[j / gs_w];
		ival = 0;
	}
	return val;
}

void linear_q(float *xout, QuantizedTensor * x, int8_t * p, float *sf,
	      LinearConfigQ lc, bool relu)
{
	// w(out,in) @ x (in,) + b(out,) -> xout (out,)
	int in = lc.in;
	int out = lc.out;
	int gs_w = lc.gs_weight;
	int gs_b = lc.gs_bias;

	int8_t *w = p + lc.qoffset;
	int8_t *b = w + in * out;
	float *sw = sf + lc.soffset;
	float *sb = sw + in * out / gs_w;
	for (int i = 0; i < out; i++) {
		float val = linear_q_matmul(x, w, sw, in, gs_w, i);
		float bias_val =
		    gs_b > 0 ? ((float) b[i]) * sb[i / gs_b] : 0.0f;
		xout[i] = relu ? fmax(val + bias_val, 0.0f) : val + bias_val;
	}
}

static inline float im2col_get_pixel(float *im, int height, int width, int row,
				     int col, int channel, int pad)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[col + width * (row + height * channel)];
}

static void im2col_generic(float *col, float *im, int *height, int *width,
			   int nchannels, int ksize, int stride, int pad)
{
	// im (nchannels, height, width) -> col (col_size, out_height * out_width)
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

	int col_size = nchannels * ksize * ksize;
	for (int c = 0; c < col_size; c++) {
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int chan = c / ksize / ksize;
		for (int h = 0; h < out_height; h++) {
			for (int w = 0; w < out_width; w++) {
				int in_row = h_offset + h * stride;
				int in_col = w_offset + w * stride;
				int col_index =
				    (c * out_height + h) * out_width + w;
				col[col_index] =
				    im2col_get_pixel(im, *height, *width,
						     in_row, in_col, chan, pad);
			}
		}
	}
	// update current height and width
	*height = out_height;
	*width = out_width;
}

void im2col(float *col, float *im, int *height, int *width, ConvConfig cc)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad);
}

void im2col_q(float *col, float *im, int *height, int *width, ConvConfigQ cc)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad);
}

void conv(float *xout, float *x, float *p, ConvConfig cc, int out,
		 bool relu)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(nchannels,) -> xout (nchannels,out)
	int nchannels = cc.oc;
	int in = cc.ic * cc.ksize * cc.ksize;

	float *w = p + cc.offset;
	float *b = w + nchannels * in;
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val = 0.0f;
			for (int j = 0; j < in; j++) {
				val += w[c * in + j] * x[j * out + i];
			}
			float bias_val = cc.bias ? b[c] : 0.0f;
			xout[c * out + i] =
			    relu ? fmax(val + bias_val, 0.0f) : val + bias_val;
		}
	}
}

static float conv_q_matmul(QuantizedTensor * x, int8_t * w, float *sw,
			   int in, int out, int gs_w, int c, int i)
{
	float val = 0.0f;
	// do the matmul in groups of gs_w
	for (int j = 0; j < in; j += gs_w) {
		int32_t ival = 0;
		for (int k = 0; k < gs_w; k++) {
			ival +=
			    ((int32_t) x-> q[(j + k) * out + i]) *
			    ((int32_t) w[c * in + (j + k)]);
		}
		val +=
		    (float) ival *x->s[(i * in + j) / gs_w] *
		    sw[(c * in + j) / gs_w];
	}
	return val;
}

void conv_q(float *xout, QuantizedTensor * x, int8_t * p,
		   float *sf, ConvConfigQ cc, int out, bool relu)
{
	// w (nchannels,1,in) @ x (1,in,out) + b(nchannels,) -> xout (nchannels,out)
	int nchannels = cc.oc;
	int in = cc.ic * cc.ksize * cc.ksize;
	int gs_w = cc.gs_weight;
	int gs_b = cc.gs_bias;

	int8_t *w = p + cc.qoffset;
	int8_t *b = w + in * out;
	float *sw = sf + cc.soffset;
	float *sb = sw + in * out / gs_w;
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < out; i++) {
			float val =
			    conv_q_matmul(x, w, sw, in, out, gs_w, c, i);
			float bias_val =
			    gs_b > 0 ? ((float) b[i]) * sb[i / gs_b] : 0.0f;
			xout[c * out + i] =
			    relu ? fmax(val + bias_val, 0.0f) : val + bias_val;
		}
	}
}

void batchnorm(float *xout, float *x, float *p, int nchannels, int in,
	       bool relu)
{
	// x (nchannels,in) -> xout (nchannels,in)
	float *w = p;
	float *b = p + nchannels;
	float *running_mean = p + 2 * nchannels;
	float *running_var = p + 3 * nchannels;

	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < in; i++) {
			float val =
			    (x[c * in + i] -
			     running_mean[c]) / sqrt(running_var[c] +
						     eps) * w[c] + b[c];
			xout[c * in + i] = (relu) ? fmax(val, 0.0f) : val;
		}
	}
}

typedef float (*PoolOperation)(float, float);

static inline float pool_get_max(float inp, float val)
{
	return fmax(inp, val);
}

static inline float pool_add(float inp, float val)
{
	return val + inp;
}

static inline float pool_get_pixel(float *x, int height, int width, int ksize,
				   int in_start_row, int in_start_col,
				   PoolOperation op, int c)
{
	float val = 0.0f;
	for (int k = 0; k < ksize * ksize; k++) {
		int in_row = in_start_row + k / ksize;
		int in_col = in_start_col + k % ksize;
		if (in_row >= 0 && in_row < height && in_col >= 0
		    && in_col < width) {
			float inp =
			    x[c * height * width + in_row * width + in_col];
			val = op(inp, val);
		}
	}
	return val;
}

static void pool_generic(float *xout, float *x, int *height, int *width,
			int nchannels, int ksize, int stride, int pad,
			PoolOperation op)
{
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;
	int out_size = out_height * out_width;

	for (int c = 0; c < nchannels; c++) {
		for (int pixel = 0; pixel < out_size; pixel++) {
			int out_row = pixel / out_width;
			int out_col = pixel % out_width;
			int in_start_row = out_row * stride - pad;
			int in_start_col = out_col * stride - pad;

			float val =
			    pool_get_pixel(x, *height, *width, ksize,
					   in_start_row, in_start_col, op, c);
			xout[c * out_size + pixel] = val;
		}
	}
	*height = out_height;
	*width = out_width;
}

void maxpool(float *xout, float *x, int *height, int *width, int nchannels,
	     int ksize, int stride, int pad)
{
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_get_max);
}

void avgpool(float *xout, float *x, int *height, int *width, int nchannels,
	     int ksize, int stride, int pad)
{
	pool_generic(xout, x, height, width, nchannels, ksize, stride, pad,
		     pool_add);
	for (int i = 0; i < nchannels * (*height) * (*width); i++) {
		xout[i] /= ksize * ksize;
	}
}

void matadd(float *x, float *y, int size)
{
	for (int i = 0; i < size; i++) {
		x[i] = x[i] + y[i];
	}
}

void relu(float *x, int size)
{
	// apply ReLU (Rectified Linear Unit) activation
	for (int i = 0; i < size; i++) {
		x[i] = fmax(0.0f, x[i]);
	}
}

void softmax(float *x, int size)
{
	// find max value (for numerical stability)
	float max_val = x[0];
	for (int i = 1; i < size; i++) {
		if (x[i] > max_val) {
			max_val = x[i];
		}
	}
	// exp and sum
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		x[i] = expf(x[i] - max_val);
		sum += x[i];
	}
	// normalize
	for (int i = 0; i < size; i++) {
		x[i] /= sum;
	}
}

static inline float quantize_find_wmax(float *x, int gs, int a, int b)
{
	// find the maximum absolute value in the current group
	float wmax = 0.0;
	for (int i = 0; i < gs; i++) {
		float val =
			fabs(x[a * i + b]);
		if (val > wmax) {
			wmax = val;
		}
	}
	return wmax;
}

static void quantize_scale(QuantizedTensor * qx, float *x, int gs, int a,
			   int b, float scale)
{
	// scale and save weights
	for (int i = 0; i < gs; i++) {
		float quant_value = x[a * i + b] / scale;
		int8_t quantized = (int8_t) round(quant_value);
		qx->q[a * i + b] = quantized;
	}
}

void quantize(QuantizedTensor * qx, float *x, int n, int gs)
{
	int num_groups = n / gs;
	float Q_MAX = 127.0f;

	for (int group = 0; group < num_groups; group++) {
		int idx = group * gs; // start idx
		float wmax = quantize_find_wmax(x, gs, 1, idx);
		float scale = wmax / Q_MAX;
		qx->s[group] = scale;
		quantize_scale(qx, x, gs, 1, idx, scale);
	}
}

void quantize2d(QuantizedTensor * qx, float *x, ConvConfigQ cc, int ncols)
{
	int nrows = cc.ic * cc.ksize * cc.ksize;
	int gs = cc.gs_weight;
	int num_groups = nrows / gs;
	float Q_MAX = 127.0f;

	for (int col = 0; col < ncols; col++) {
		for (int group = 0; group < num_groups; group++) {
			int idx = group * gs * ncols + col; // start idx
			float wmax = quantize_find_wmax(x, gs, ncols, idx);
			float scale = wmax / Q_MAX;
			qx->s[col * num_groups + group] = scale;
			quantize_scale(qx, x, gs, ncols, idx, scale);
		}
	}
}

void dequantize(QuantizedTensor * qx, float *x, int n, int gs)
{
	for (int i = 0; i < n; i++) {
		x[i] = qx->q[i] * qx->s[i / gs];
	}
}

void dequantize2d(QuantizedTensor * qx, float *x, int nrows, int ncols, int gs)
{
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			x[i * ncols + j] =
			    qx->q[i * ncols + j] * qx->s[(j * nrows + i) / gs];
		}
	}
}
