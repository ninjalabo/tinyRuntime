#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "properties.h"

// avoid division by zero
#define eps 0.00001f

int batch_size = 1;

// TODO: optimize batch processing in all func*.c modules

#define MAKE_UINT32(u0, u1, u2, u3) \
    ((uint32_t)(u0) | ((uint32_t)(u1) << 8) | ((uint32_t)(u2) << 16) | ((uint32_t)(u3) << 24))

#define PRECISION_BITS (32 - 8 - 2)

struct filter {
    double (*filter)(double x);
    double support;
};

static inline double
bilinear_filter(double x) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return 1.0 - x;
    }
    return 0.0;
}

static struct filter BILINEAR = {bilinear_filter, 1.0};

static inline uint8_t
clip8(int in) {
    return (uint8_t)(in < 0 ? 0 : (in > 255 ? 255 : in));
}

int
precompute_coeffs(
    int inSize,
    float in0,
    float in1,
    int outSize,
    struct filter *filterp,
    int **boundsp,
    double **kkp) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, ksize, xmin, xmax;
    int *bounds;
    double *kk, *k;

    /* prepare for horizontal stretch */
    filterscale = scale = (double)(in1 - in0) / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    /* determine support size (length of resampling filter) */
    support = filterp->support * filterscale;

    /* maximum number of coeffs */
    ksize = (int)ceil(support) * 2 + 1;

    /* coefficient buffer */
    kk = (double*) malloc(outSize * ksize * sizeof(double));
    bounds = (int*) malloc(outSize * 2 * sizeof(int));

    for (xx = 0; xx < outSize; xx++) {
        center = in0 + (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        // Round the value
        xmin = (int)(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        // Round the value
        xmax = (int)(center + support + 0.5);
        if (xmax > inSize) {
            xmax = inSize;
        }
        xmax -= xmin;
        k = &kk[xx * ksize];
        for (x = 0; x < xmax; x++) {
            double w = filterp->filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0) {
                k[x] /= ww;
            }
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < ksize; x++) {
            k[x] = 0;
        }
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    *boundsp = bounds;
    *kkp = kk;
    return ksize;
}

void
normalize_coeffs_8bpc(int outSize, int ksize, double *prekk) {
    int x;
    int32_t *kk;

    // use the same buffer for normalized coefficients
    kk = (int32_t *)prekk;

    for (x = 0; x < outSize * ksize; x++) {
        if (prekk[x] < 0) {
            kk[x] = (int)(-0.5 + prekk[x] * (1 << PRECISION_BITS));
        } else {
            kk[x] = (int)(0.5 + prekk[x] * (1 << PRECISION_BITS));
        }
    }
}

void
ImagingResampleHorizontal_8bpc(
    uint8_t *imOut, uint8_t *imIn, int nch, int w1, int w0, int h0, int offset, int ksize, int *bounds, double *prekk) {
    int ss0, ss1, ss2;
    int xx, yy, x, xmin, xmax;
    int32_t *k, *kk;

    // use the same buffer for normalized coefficients
    kk = (int32_t*) prekk;
    normalize_coeffs_8bpc(w1, ksize, prekk);

    if (nch == 1) {
        for (yy = 0; yy < h0; yy++) {
            for (xx = 0; xx < w1; xx++) {
                xmin = bounds[xx * 2 + 0];
                xmax = bounds[xx * 2 + 1];
                k = &kk[xx * ksize];
                ss0 = 1 << (PRECISION_BITS - 1);
                for (x = 0; x < xmax; x++) {
                    ss0 += imIn[(yy + offset)*w0 + x + xmin] * k[x];
                }
                imOut[yy*w1 + xx] = clip8(ss0);
            }
        }
    } else {
	for (yy = 0; yy < h0; yy++) {
		for (xx = 0; xx < w1; xx++) {
			uint32_t v;
			xmin = bounds[xx * 2 + 0];
			xmax = bounds[xx * 2 + 1];
			k = &kk[xx * ksize];
			ss0 = ss1 = ss2 = 1 << (PRECISION_BITS - 1);
			for (x = 0; x < xmax; x++) {
				ss0 +=
				    imIn[(yy + offset)*w0 + (x + xmin)*4 + 0] *
				    k[x];
				ss1 +=
				    imIn[(yy + offset)*w0 + (x + xmin)*4 + 1] *
				    k[x];
				ss2 +=
				    imIn[(yy + offset)*w0 + (x + xmin)*4 + 2] *
				    k[x];
			}
			v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), 0);
			memcpy(&imOut[yy*w1 + xx * sizeof(v)], &v, sizeof(v));
		}
	}
    }
}

void
ImagingResampleVertical_8bpc(
    uint8_t *imOut, uint8_t *imIn, int nch, int w1, int h1, int w0, int h0, int offset, int ksize, int *bounds, double *prekk) {
    int ss0, ss1, ss2;
    int xx, yy, y, ymin, ymax;
    int32_t *k, *kk;

    // use the same buffer for normalized coefficients
    kk = (int32_t*) prekk;
    normalize_coeffs_8bpc(h1, ksize, prekk);

    if (nch == 1) {
        for (yy = 0; yy < h1; yy++) {
            k = &kk[yy * ksize];
            ymin = bounds[yy * 2 + 0];
            ymax = bounds[yy * 2 + 1];
            for (xx = 0; xx < w1; xx++) {
                ss0 = 1 << (PRECISION_BITS - 1);
                for (y = 0; y < ymax; y++) {
		    ss0 += imIn[(y + ymin)*w0 + xx] * k[y];
                }
		imOut[yy*w1 + xx] = clip8(ss0);
            }
        }
    } else {
            for (yy = 0; yy < h1; yy++) {
                k = &kk[yy * ksize];
                ymin = bounds[yy * 2 + 0];
                ymax = bounds[yy * 2 + 1];
                for (xx = 0; xx < w1; xx++) {
                    uint32_t v;
                    ss0 = ss1 = ss2 = 1 << (PRECISION_BITS - 1);
                    for (y = 0; y < ymax; y++) {
			ss0 += imIn[(y + ymin)*w0 + xx*4 + 0] * k[y];
			ss1 += imIn[(y + ymin)*w0 + xx*4 + 1] * k[y];
			ss2 += imIn[(y + ymin)*w0 + xx*4 + 2] * k[y];
                    }
                    v = MAKE_UINT32(clip8(ss0), clip8(ss1), clip8(ss2), 0);
		    memcpy(&imOut[yy*w1 + xx * sizeof(v)], &v, sizeof(v));
                }
            }
    }
}

void
ImagingResampleInner(
    uint8_t *imOut,
    uint8_t *imIn,
    int nch,
    int w1, int h1,
    float box[4]) {

    struct filter *filterp = &BILINEAR;
    uint8_t *imTemp = NULL;
    int i, need_horizontal, need_vertical;
    int ybox_first;
    int ksize_horiz, ksize_vert;
    int *bounds_horiz, *bounds_vert;
    double *kk_horiz, *kk_vert;

    int h0 = box[3] - box[1];
    int w0 = box[2] - box[0];

    need_horizontal = w1 != w0 || box[0] || box[2] != w1;
    need_vertical = h1 != h0 || box[1] || box[3] != h1;

    ksize_horiz = precompute_coeffs(
        w0, box[0], box[2], w1, filterp, &bounds_horiz, &kk_horiz);
    ksize_vert = precompute_coeffs(
        h0, box[1], box[3], h1, filterp, &bounds_vert, &kk_vert);

    // First used row in the source image
    ybox_first = bounds_vert[0];

    /* two-pass resize, horizontal pass */
    if (need_horizontal) {
        // Shift bounds for vertical pass
        for (i = 0; i < h1; i++) {
            bounds_vert[i * 2] -= ybox_first;
        }
	imTemp = (uint8_t*) malloc(w1 * h0 * nch * sizeof(uint8_t));
	ImagingResampleHorizontal_8bpc(imTemp, imIn, nch, w1, w0, h0, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
        free(bounds_horiz);
        free(kk_horiz);
        memcpy(imIn, imTemp, w1 * h0 * nch * sizeof(uint8_t));
	free(imTemp);
    } else {
        // Free in any case
        free(bounds_horiz);
        free(kk_horiz);
    }
    FILE *f = fopen("horizontal.txt", "w");
    for (int i = 0; i < w1 * h0 * nch; i++) {
	fprintf(f, "%hhu\n", imIn[i]);
    }
    fclose(f);

    /* vertical pass */
    if (need_vertical) {
	/* imIn can be the original image or horizontally resampled one */
	ImagingResampleVertical_8bpc(imOut, imIn, nch, w1, h1, w1, h0, 0, ksize_vert, bounds_vert, kk_vert);
        free(bounds_vert);
        free(kk_vert);
    } else {
        // Free in any case
        free(bounds_vert);
        free(kk_vert);
    }

    if (!need_vertical && !need_horizontal) {
        memcpy(imOut, imIn, w0 * h0 * nch * sizeof(uint8_t));
    }
}

void load_imagenette(float *transformed, char *filename, int h1, int w1)
{
	// load image
	int nch0, h0, w0; // image shape before resizing
	int nch1 = 3; // number of channels
	uint8_t *data =
		stbi_load(filename, &w0, &h0, &nch0, nch1);
	if (!data) {
		fprintf(stderr, "Error loading image: %s\n", filename);
		exit(EXIT_FAILURE);
	}
	// uint8_t *image = (uint8_t*) malloc(3 * h0 * w0 * sizeof(float));
	// for (int c = 0; c < nch1; c++){
	// 	for (int h = 0; h < h0; h++) {
	// 		for (int w = 0; w < w0; w++) {
	// 			int dst_idx = c*h0*w0 + h*w0 + w;
	// 			int src_idx = h*w0*nch0 + w*nch0 + c;
	// 			image[dst_idx] = data[src_idx];
	// 		}
	// 	}
	// }
	// stbi_image_free(data);
	
	float m =
	    (float) w0 / w1 < (float) h0 / h1 ? (float) w0 / w1 :
	    (float) h0 / h1;
	int crop_size[2] = { (int) (w1 * m), (int) (h1 * m) };
	int tl[2] =
	    { (int) ((w0 - crop_size[0]) * 0.5f),
	    (int) ((h0 - crop_size[1]) * 0.5f) };
	// crop image
	uint8_t *cropped =
	    (uint8_t*) malloc(nch1 * crop_size[0] * crop_size[1] * sizeof(uint8_t));
	if (tl[0] >= 0 || tl[1] >= 0 || tl[0] + crop_size[0] <= w0
	    || tl[1] + crop_size[1] <= h0) {
		int tl_tmp[2] =
		    { tl[0] < 0 ? 0 : tl[0], tl[1] < 0 ? 0 : tl[1] };
		// memcpy(cropped, data + (tl_tmp[1] * w0 + tl_tmp[0]) * nch1,
		// 	nch1 * crop_size[0] * crop_size[1] * sizeof(uint8_t));
		for (int h = 0; h < crop_size[1]; h++) {
			memcpy(cropped + h * crop_size[0] * nch1,
			       data + (h + tl_tmp[1]) * w0 * nch1 + tl_tmp[0] * nch1,
			       crop_size[0] * nch1 * sizeof(uint8_t));
		}
		// for (int c = 0; c < nch1; c++) {
		// 	for (int h = 0; h < crop_size[1]; h++) {
		// 		for (int w = 0; w < crop_size[0]; w++) {
		// 			int dst_idx =
		// 			    c * crop_size[0] * crop_size[1] +
		// 			    h * crop_size[0] + w;
		// 			int src_idx1 = h + tl_tmp[1];
		// 			int src_idx2 = w + tl_tmp[0];
		// 			// if (src_idx1 >= h0 || src_idx2 >= w0) {
		// 			// 	resized[dst_idx] = 0.0f;
		// 			// 	continue;
		// 			// }
		// 			int src_idx =
		// 			    c * h0 * w0 + src_idx1 * w0 + 
		// 			    src_idx2;
		// 			cropped[dst_idx] = image[src_idx];
		// 		}
		// 	}
		// }
	}
	// pad image
	// if (tl[0] < 0 || tl[1] < 0 || tl[0] + crop_size[0] > w0 ||
	//     tl[1] + crop_size[1] > h0) {
	// 	// border index (left, top, right, bottom)
	// 	int paddings[4] =
	// 	    { tl[0] < 0 ? -tl[0] : 0, tl[1] < 0 ? -tl[1] : 0,
	// 	    (crop_size[0] - w0) + tl[0] > 0 ?
	// 	    (crop_size[0] - w0) + tl[0] : 0,
	// 	    (crop_size[1] - h0) + tl[1] > 0 ?
	// 	    (crop_size[1] - h0) + tl[1] : 0 };
	// 	float *padded =
	// 	    (float*) malloc(nch1 * crop_size[0] * crop_size[1] *
	// 	    sizeof(float));
	// 	for (int c = 0; c < nch1; c++) {
	// 		for (int h = 0; h < crop_size[1]; h++) {
	// 			for (int w = 0; w < crop_size[0]; w++) {
	// 				int dst_idx =
	// 				    c * crop_size[0] * crop_size[1] +
	// 				    h * crop_size[0] + w;
	// 				int src_idx =
	// 				    c * h0 * w0 +
	// 				    (h + tl[1] - paddings[1]) * w0 +
	// 				    w + tl[0] - paddings[0];
	// 				if (src_idx < 0 || src_idx >= nch1 * h0 * w0) {
	// 					resized[dst_idx] = 0.0f;
	// 				} else {
	// 					resized[dst_idx] = image[src_idx];
	// 				}
	// 			}
	// 		}
	// 	}

	// }
	// resize image by bilinear interpolation
	stbi_image_free(data);
	uint8_t *resized = (uint8_t*) malloc(nch1 * h1 * w1 * sizeof(float));
	ImagingResampleInner(resized, cropped, nch1, w1, h1, (float[4]) { 0.0f, 0.0f, (float) crop_size[0], (float) crop_size[1] });
	FILE *f = fopen("cropped.txt", "w");
	for (int i = 0; i < nch1 * crop_size[0] * crop_size[1]; i++) {
		fprintf(f, "%hhu\n", cropped[i]);
	}
	fclose(f);
	f = fopen("resized.txt", "w");
	for (int i = 0; i < nch1 * h1 * w1; i++) {
		fprintf(f, "%hhu\n", resized[i]);
	}
	fclose(f);
	// normalize
	// stats from ImageNet dataset
	float mean[] = { 0.485, 0.456, 0.406 };
	float std[] = { 0.229, 0.224, 0.225 };
	for (int c = 0; c < nch1; c++){
		for (int h = 0; h < h1; h++) {
			for (int w = 0; w < w1; w++) {
				int dst_idx = c*h1*w1 + h*w1 + w;
				int src_idx = h*w1*nch1 + w*nch1 + c;
				transformed[dst_idx] =
				    (resized[src_idx] / 255.0f - mean[c]) /
				    std[c];
			}
		}
	}
	free(cropped);
	free(resized);
}


static float im2col_get_pixel(float *im, int height, int width, int row,
			      int col, int channel, int pad, int start_idx)
{
	row -= pad;
	col -= pad;
	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[start_idx + col + width * (row + height * channel)];
}

static void im2col_populate_column(float *col, float *im, int height, int width,
				   int out_height, int out_width, int ksize,
				   int stride, int pad, int c, int col_size,
				   int col_idx, int im_idx)
{
	int w_offset = c % ksize;
	int h_offset = (c / ksize) % ksize;
	int chan = c / ksize / ksize;
	for (int h = 0; h < out_height; h++) {
		for (int w = 0; w < out_width; w++) {
			int in_row = h_offset + h * stride;
			int in_col = w_offset + w * stride;
			int idx = (h * out_width + w) * col_size + c;
			col[col_idx + idx] =
			    im2col_get_pixel(im, height, width, in_row, in_col,
					     chan, pad, im_idx);
		}
	}
}

static void im2col_generic(float *col, float *im, int *height, int *width,
			   int nchannels, int ksize, int stride, int pad)
{
	// im (nchannels, height, width) -> col (out_height * out_width, col_size)
	int out_height = (*height + 2 * pad - ksize) / stride + 1;
	int out_width = (*width + 2 * pad - ksize) / stride + 1;

	int col_size = nchannels * ksize * ksize;
	for (int bs = 0; bs < batch_size; bs++) {
		int col_idx = bs * col_size * out_height * out_width;
		int im_idx = bs * nchannels * (*height) * (*width);
		for (int c = 0; c < col_size; c++) {
			im2col_populate_column(col, im, *height, *width,
					       out_height, out_width, ksize,
					       stride, pad, c, col_size,
					       col_idx, im_idx);
		}
	}
	// update current height and width
	*height = out_height;
	*width = out_width;
}

void im2col(float *col, float *im, ConvConfig cc, int *height, int *width)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad);
}

void im2col_q(float *col, float *im, ConvConfigQ cc, int *height, int *width)
{
	im2col_generic(col, im, height, width, cc.ic, cc.ksize, cc.stride,
		       cc.pad);
}

void batchnorm_one_group(float *xout, float *x, float *w, float *b,
			 float *rmean, float *rvar, int group, int size,
			 int start_idx)
{
	for (int i = 0; i < size; i++) {
		float val =
		    (x[start_idx + group * size + i] - rmean[group]) /
		    sqrt(rvar[group] + eps) * w[group] + b[group];
		xout[start_idx + group * size + i] = val;
	}
}
// TODO: optimize using blis and/or combine parameters with previous layer
// TODO: make batchnorm1d and batchnorm2d if that is better
void batchnorm(float *xout, float *x, float *p, BnConfig bc, int size)
{
	// x (ngroups,size) -> xout (ngroups,size)
	int ngroups = bc.ic;
	float *w = p + bc.offset;
	float *b = w + ngroups;
	float *rmean = b + ngroups;
	float *rvar = rmean + ngroups;

	for (int bs = 0; bs < batch_size; bs++) {
		int start_idx = bs * ngroups * size;
		for (int group = 0; group < ngroups; group++) {
			batchnorm_one_group(xout, x, w, b, rmean, rvar, group,
					    size, start_idx);
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

static float pool_get_pixel(float *x, int height, int width, int ksize,
			    int in_start_row, int in_start_col,
			    PoolOperation op, int group)
{
	float val = 0.0f;
	for (int k = 0; k < ksize * ksize; k++) {
		int in_row = in_start_row + k / ksize;
		int in_col = in_start_col + k % ksize;
		if (in_row >= 0 && in_row < height && in_col >= 0
		    && in_col < width) {
			float inp =
			    x[group * height * width + in_row * width + in_col];
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

	for (int group = 0; group < batch_size * nchannels; group++) {
		for (int pixel = 0; pixel < out_size; pixel++) {
			int out_row = pixel / out_width;
			int out_col = pixel % out_width;
			int in_start_row = out_row * stride - pad;
			int in_start_col = out_col * stride - pad;

			float val =
			    pool_get_pixel(x, *height, *width, ksize,
					   in_start_row, in_start_col, op,
					   group);
			xout[group * out_size + pixel] = val;
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
	for (int i = 0; i < batch_size * nchannels * (*height) * (*width); i++)
		xout[i] /= (ksize * ksize);
}

void relu(float *x, int size)
{
	// apply ReLU (Rectified Linear Unit) activation
	for (int i = 0; i < batch_size * size; i++) {
		x[i] = x[i] > 0.0f ? x[i] : 0.0f;
	}
}

void softmax(float *x, int size)
{
	for (int bs = 0; bs < batch_size; bs++) {
		int start_idx = bs * size;
		// find max value (for numerical stability)
		float max_val = x[0];
		for (int i = 1; i < size; i++) {
			if (x[start_idx + i] > max_val) {
				max_val = x[start_idx + i];
			}
		}
		// exp and sum
		float sum = 0.0f;
		for (int i = 0; i < size; i++) {
			x[start_idx + i] = expf(x[start_idx + i] - max_val);
			sum += x[start_idx + i];
		}
		// normalize
		for (int i = 0; i < size; i++) {
			x[start_idx + i] /= sum;
		}
	}
}

void find_max(int *xout, float *x, int nclasses)
{
	for (int bs = 0; bs < batch_size; bs++) {
		float cmax = 0.0f;
		int max_idx;
		for (int i = 0; i < nclasses; i++) {
			if (cmax < x[bs * nclasses + i]) {
				cmax = x[bs * nclasses + i];
				max_idx = i;
			}
		}
		xout[bs] = max_idx;
	}
}

void matcopy_float(float *xout, float *x, int size) {
	memcpy(xout, x, batch_size * size * sizeof(float));
}
