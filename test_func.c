#include "CppUTest/TestHarness.h"

#include <math.h>
#include <string.h>

#include "func_common.h"
#include "func.h"
#include "func_q.h"

#define TOL 0.00001
#define eps 0.00001f
#define Q_MAX 127.0f
#define Q_MIN -128.0f
#define Q_RANGE 255.0f

TEST_GROUP(TestGroup)
{
        void setup() {
		batch_size = 2;
        }
        void teardown() {
        }
};

#ifdef USE_DQ_FUNC // func.c and func_dq.c

TEST(TestGroup, ReadImagenette)
{
	int nch = 3, h = 224, w = 224;
	int img_sz = nch * h * w;
	float ref[batch_size * img_sz];
	for (int i = 0; i < batch_size * img_sz; i++) {
		ref[i] = i;
	}
	const char *paths[] =
	    { "test_read_imagenette1.bin", "test_read_imagenette2.bin" };
	FILE *file = fopen(paths[0], "wb");
	fwrite(ref, sizeof(float), img_sz, file);
	fclose(file);
	file = fopen(paths[1], "wb");
	fwrite(&ref[img_sz], sizeof(float), img_sz, file);
	fclose(file);

	float images[batch_size * img_sz];
	read_imagenette_image((char**) paths, images, batch_size);
	MEMCMP_EQUAL(ref, images, batch_size * img_sz * sizeof(float));

	remove(paths[0]);
	remove(paths[1]);
}

TEST(TestGroup, Linear)
{
	// vanilla
        int in = 2, out = 2, offset = 1;
	int gs_w = 36, gs_b = 8;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float p[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f };
        LinearConfig lc = { in, out, offset, 1 };

        float xout[batch_size * out];
        linear(xout, x, p, lc);

        float ref[] = { 6.0f, 13.0f, 12.0f, 27.0f };
        for (int i = 0; i < batch_size * out; i++) {
                DOUBLES_EQUAL(ref[i], xout[i], TOL);
        }
	// quantized with out zero point
	in = 72, out = 8;
        int size_q = in * out + out + offset;
        int size_s = in * out / gs_w + out / gs_b + offset;
        int8_t qx_q[batch_size * in];
        float qx_sf[batch_size * in / gs_w];
        for (int i = 0; i < batch_size * in; i++)
		qx_q[i] = i / 18;
        for (int i = 0; i < batch_size * in / gs_w; i++)
		qx_sf[i] = i;
        QuantizedTensor qx = {.q = qx_q, .scale = qx_sf, .zero_point = NULL};
        int8_t pq[size_q];
        float sf[size_s];
        for (int i = 0; i < size_q; i++)
		pq[i] = (i - 1) / 18;
        for (int i = 0; i < size_s; i++)
		sf[i] = i - 1;
        LinearConfigQ lc_q = { in, out, offset, offset, gs_w, gs_b };

        float xout_q[batch_size * out];
        linear_q(xout_q, &qx, pq, sf, lc_q);

	float ref_q[] =
	    { 746.0f, 2294.0f, 5282.0f, 9710.0f, 15578.0f, 22886.0f, 31634.0f,
	    41822.0f, 2294.0f, 17234.0f, 48590.0f, 96362.0f, 160550.0f,
	    241154.0f, 338174.0f, 451610.0f };
        for (int i = 0; i < batch_size * out; i++) {
                DOUBLES_EQUAL(ref_q[i], xout_q[i], TOL);
        }
	// quantized with zero point
	int qx_zp[batch_size * in / gs_w];
	for (int i = 0; i < batch_size * in / gs_w; i++)
		qx_zp[i] = 1;
	qx.zero_point = qx_zp;
	// fill first half sf_zp by scales and zero points
	int w_sf_size = in * out / gs_w;
	int b_sf_size = out / gs_b;
	int size_sf_zp = offset + 2 * (w_sf_size + b_sf_size);
	float sf_zp[size_sf_zp];
	int *zp = (int *) sf_zp;
        for (int i = 0; i < offset + w_sf_size; i++) {
		sf_zp[i] = i - 1; // scaling factor of weights
		zp[i + w_sf_size] = 1; // zero point of weights
	}
	int start_idx = 2 * w_sf_size + offset;
	for (int i = start_idx; i < start_idx + b_sf_size; i++) {
		sf_zp[i] = 16; // scaling factor of biases
		zp[i + b_sf_size] = 1; // zero point of biases
	}
	linear_q(xout_q, &qx, pq, sf_zp, lc_q);
	float ref_q2[] =
	    { 586.0f, 1414.0f, 3106.0f, 5662.0f, 9082.0f, 13366.0f, 18514.0f,
	    24526.0f, 1414.0f, 12178.0f, 36478.0f, 74314.0f, 125686.0f,
	    190594.0f, 269038.0f, 361018.0f };
	for (int i = 0; i < batch_size * out; i++) {
                DOUBLES_EQUAL(ref_q2[i], xout_q[i], TOL);
        }
}

TEST(TestGroup, Im2col)
{
        int h = 2, w = 2, hq = 2, wq = 2;
        int ksize = 2, stride = 2, pad = 1, ic = 2;
        int col_size = ksize * ksize * ic * h * w;

        float im[] =
	    { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
	    9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
        ConvConfig cc = { ksize, stride, pad, ic, 0, 0, 0 };	// 0 doesn't affect
        ConvConfigQ cc_q = { ksize, stride, pad, ic, 0, 0, 0, 0, 0 };

        float col[batch_size * col_size];
        im2col(col, im, cc, &h, &w);

        float col_q[batch_size * col_size];
        im2col_q(col_q, im, cc_q, &hq, &wq);

        float ref[] = {
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 5.0f,
		0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f,
		0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 7.0f, 0.0f, 0.0f,
		4.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f, 13.0f,
		0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 14.0f, 0.0f,
		0.0f, 11.0f, 0.0f, 0.0f, 0.0f, 15.0f, 0.0f, 0.0f,
		12.0f, 0.0f, 0.0f, 0.0f, 16.0f, 0.0f, 0.0f, 0.0f,
        };
        MEMCMP_EQUAL(ref, col, batch_size * col_size * sizeof(float));
        MEMCMP_EQUAL(ref, col_q, batch_size * col_size * sizeof(float));
        DOUBLES_EQUAL(h, hq, TOL);
        DOUBLES_EQUAL(w, wq, TOL);
}

TEST(TestGroup, Conv)
{
	// vanilla
        int h = 1, w = 1;
        int ksize = 2, stride = 1, pad = 1, ic = 1, oc = 2;
	int offset = 1, bias = 1;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        float p[] =
            { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f,
            2.0f };
        ConvConfig cc = { ksize, stride, pad, ic, oc, offset, bias };

        float xout[batch_size * oc * h * w];
        conv(xout, x, p, cc, h, w);

        float ref[] = { 31.0f, 72.0f, 71.0f, 176.0f };
        for (int i = 0; i < batch_size * oc * h * w; i++) {
                DOUBLES_EQUAL(ref[i], xout[i], TOL);
        }
	// quantized without zero point
	int gs_w = 40, gs_b = 2;
	ic = 20, h = 3, w = 3;
        int nrows = ksize * ksize * ic;
	int size_x = batch_size * nrows * h * w;
        int size_q = nrows * oc + oc + offset;
        int size_s = nrows * oc / gs_w + oc / gs_b + offset;
        int8_t qx_q[size_x];
        float qx_sf[size_x / gs_w];
        for (int i = 0; i < size_x; i++)
		qx_q[i] = i / 20;
        for (int i = 0; i < size_x / gs_w; i++)
		qx_sf[i] = i;
        QuantizedTensor qx = {.q = qx_q, .scale = qx_sf, .zero_point = NULL};
        int8_t pq[size_q];
        float sf[size_s];
        for (int i = 0; i < size_q; i++)
		pq[i] = (i - 1) / 20;
        for (int i = 0; i < size_s; i++)
		sf[i] = i - 1;
        ConvConfigQ cc_q =
            { ksize, stride, pad, ic, oc, offset, offset, gs_w, gs_b };

        float xout_q[size_x];
        conv_q(xout_q, &qx, pq, sf, cc_q, h, w);

	float ref_q[] =
	    { 292.0f, 2012.0f, 5332.0f, 10252.0f, 16772.0f, 24892.0f, 34612.0f,
	    45932.0f, 58852.0f, 2012.0f, 18612.0f, 53452.0f, 106532.0f,
	    177852.0f, 267412.0f, 375212.0f, 501252.0f, 645532.0f, 73372.0f,
	    89492.0f, 107212.0f, 126532.0f, 147452.0f, 169972.0f,194092.0f,
	    219812.0f, 247132.0f, 808052.0f, 988812.0f, 1187812.0f, 1405052.0f,
	    1640532.0f, 1894252.0f, 2166212.0f, 2456412.0f, 2764852.0f };
        for (int i = 0; i < batch_size * oc * h * w; i++) {
                DOUBLES_EQUAL(ref_q[i], xout_q[i], TOL);
        }
	// quantized with zero point
	int qx_zp[size_x / gs_w];
	for (int i = 0; i < size_x / gs_w; i++)
		qx_zp[i] = 1;
	qx.zero_point = qx_zp;
	// fill first half sf_zp by scales and zero points
	int w_sf_size = nrows * oc / gs_w;
	int b_sf_size = oc / gs_b;
	int size_sf_zp = offset + 2 * (w_sf_size + b_sf_size);
	float sf_zp[size_sf_zp];
	int *zp = (int *) sf_zp;
        for (int i = 0; i < offset + w_sf_size; i++) {
		sf_zp[i] = i - 1; // scaling factor of weights
		zp[i + w_sf_size] = 1; // zero point of weights
	}
	int start_idx = 2 * w_sf_size + offset;
	for (int i = start_idx; i < start_idx + b_sf_size; i++) {
		sf_zp[i] = 16; // scaling factor of biases
		zp[i + b_sf_size] = 1; // zero point of biases
	}
	conv_q(xout_q, &qx, pq, sf_zp, cc_q, h, w);
	float ref_q2[] =
	    { 212.0f, 1132.0f, 3012.0f, 5852.0f, 9652.0f, 14412.0f, 20132.0f,
	    26812.0f, 34452.0f, 1132.0f, 13092.0f, 40092.0f, 82132.0f,
	    139212.0f, 211332.0f, 298492.0f, 400692.0f, 517932.0f, 43052.0f,
	    52612.0f, 63132.0f, 74612.0f, 87052.0f, 100452.0f, 114812.0f,
	    130132.0f, 146412.0f, 650212.0f, 797532.0f, 959892.0f, 1137292.0f,
	    1329732.0f, 1537212.0f, 1759732.0f, 1997292.0f, 2249892.0f };
	for (int i = 0; i < batch_size * oc * h * w; i++) {
                DOUBLES_EQUAL(ref_q2[i], xout_q[i], TOL);
        }
}

TEST(TestGroup, Batchnorm)
{
        int h = 2, w = 1, ic = 2, offset = 1;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        float p[] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 1.0f, 2.0f };
        BnConfig bc = { ic, offset };

        float xout[batch_size * ic * h * w];
        batchnorm(xout, x, p, bc, h * w);

        float ref[] =
            { 3.8999955f, 4.8999905001f, 7.9597880752f, 9.373998102f,
            7.8999755f, 8.8999705f, 13.61662818f, 15.03083821f };

        for (int i = 0; i < batch_size * ic * h * w; i++) {
                DOUBLES_EQUAL(ref[i], xout[i], TOL);
        }
}

TEST(TestGroup, Pool)
{
        int nch = 2, h_before = 1, w_before = 1;
        int ksize = 2, stride = 1, pad = 1;
        int h_after = 2, w_after = 2;
        int out_size = nch * h_after * w_after;

        float x[] = { 1.0f, -1.0f, 2.0f, -2.0f};
	float refs[4][batch_size * out_size];
	float ref_max[] =
	    { 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	    2.0f, 2.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float ref_avg[] =
            { 0.25f, 0.25f, 0.25f, 0.25f, -0.25f, -0.25f, -0.25f, -0.25f,
            0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f };
	memcpy(refs[0], ref_max, batch_size * out_size * sizeof(float));
	memcpy(refs[1], ref_avg, batch_size * out_size * sizeof(float));
	memcpy(refs[2], ref_max, batch_size * out_size * sizeof(float));
	memcpy(refs[3], ref_avg, batch_size * out_size * sizeof(float));
        int h, w;
        float xout[batch_size * out_size];
        void (*pool_functions[])(float *, float *, int *, int *, int, int, int,
                                 int) =
				     { maxpool, avgpool, maxpool_q, avgpool_q };

        for (int i = 0; i < 4; i++) {
                h = h_before;
                w = w_before;

                pool_functions[i] (xout, x, &h, &w, nch, ksize, stride, pad);
                DOUBLES_EQUAL(h_after, h, TOL);
                DOUBLES_EQUAL(w_after, w, TOL);

                for (int j = 0; j < batch_size * nch * h * w; j++) {
                        DOUBLES_EQUAL(refs[i][j], xout[j], TOL);
                }
        }
}

TEST(TestGroup, ConcatPool)
{
        int nch = 1, h = 2, w = 2;
        int ksize = 2, stride = 2, pad = 0;
	int out_h = 1, out_w = 1;

        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
	float xout[2 * batch_size * out_h * out_w];
        concat_pool(xout, x, &h, &w, nch, ksize, stride, pad);
        float ref[] = { 4.0f, 2.5f, 8.0f, 6.5f};
        for (int i = 0; i < 2 * batch_size * h * w; i++) {
            DOUBLES_EQUAL(ref[i], xout[i], TOL);
        }
}

TEST(TestGroup, Matadd)
{
        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float y[] = { 5.0f, 6.0f, 7.0f, 8.0f };
        matadd(x, y, 2);
        float ref[] = { 6.0f, 8.0f, 10.0f, 12.0f };
        for (int i = 0; i < 4; i ++) {
                DOUBLES_EQUAL(ref[i], x[i], TOL);
        }
}

TEST(TestGroup, Relu)
{
        float x[] = { 1.0f, -1.0f, 2.0f, -2.0f };
        relu(x, 2);
        float ref[] = { 1.0f, 0.0f, 2.0f, 0.0f };
        for (int i = 0; i < 4; i ++) {
                DOUBLES_EQUAL(ref[i], x[i], TOL);
        }
	float x2[] = { 1.0f, -1.0f, 2.0f, -2.0f };
	relu_q(x2, 2);
	for (int i = 0; i < 4; i++) {
		DOUBLES_EQUAL(ref[i], x[i], TOL);
	}
}

TEST(TestGroup, Softmax)
{
        float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        softmax(x, 3);
        DOUBLES_EQUAL(1.0, x[0] + x[1] + x[2], TOL);
        DOUBLES_EQUAL(1.0, x[3] + x[4] + x[5], TOL);
}

TEST(TestGroup, Matcopy)
{
	int size = 3;
	float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	float xout[batch_size * size];
	matcopy_float(xout, x, size);
	MEMCMP_EQUAL(x, xout, batch_size * size * sizeof(float));

	uint8_t qx_q[] = { 1, 2, 3, 4, 5, 6 };
	UQuantizedTensor qx = { .q = qx_q, .scale = 2.0f, .zero_point = 1 };
	uint8_t qx_out_q[batch_size * size];
	UQuantizedTensor qx_out =
	    { .q = qx_out_q, .scale = 0.0f, .zero_point = 0 };
	matcopy_uqtensor(&qx_out, &qx, size);
	MEMCMP_EQUAL(qx.q, qx_out.q, batch_size * size * sizeof(uint8_t));
}

TEST(TestGroup, Quantize)
{
	// without zero point
        int size = 4, gs = 2;
	float x[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
	int8_t qx_q[batch_size * size];
	float qx_s[batch_size * size / gs];
	QuantizedTensor qx = {.q = qx_q, .scale = qx_s, .zero_point = NULL};
	quantize(&qx, x, size, gs);
	float ref_sf[] =
	    { 2.0f / Q_MAX, 4.0f / Q_MAX, 6.0f / Q_MAX, 8.0f / Q_MAX };
	int8_t ref_q[] = { 64, 127, 95, 127, 106, 127, 111, 127 };
	for (int i = 0; i < batch_size * size / gs; i++) {
		DOUBLES_EQUAL(ref_sf[i], qx.scale[i], TOL);
	}
	MEMCMP_EQUAL(ref_q, qx.q, batch_size * size * sizeof(int8_t));
	// with zero point
	int zp[batch_size * size / gs];
	qx.zero_point = zp;
	quantize(&qx, x, size, gs);
	int8_t ref_q2[] = { -128, 127, -128, 127, -128, 127, -128, 127 };
	float ref_sf2[] =
	    { 1.0f / Q_RANGE, 1.0f / Q_RANGE, 1.0f / Q_RANGE, 1.0f / Q_RANGE};
	for (int i = 0; i < batch_size * size / gs; i++) {
		DOUBLES_EQUAL(ref_sf2[i], qx.scale[i], TOL);
	}
	MEMCMP_EQUAL(ref_q2, qx.q, batch_size * size * sizeof(int8_t));
}

#else // func_sq.c

TEST(TestGroup, Linear)
{
	int in = 2, out = 2, offset = 1, zero_point = 1, out_zero_point = 2;
	int has_bias = 1;
	float scale = 1.0f, out_scale = 0.1f;

	uint8_t qx_q[] = { 1, 2, 3, 4 };
	UQuantizedTensor qx = { .q = qx_q, .scale = 2.0f, .zero_point = 2 };
	int8_t p[offset + in * out + 4 * out];
	for (int i = 0; i < offset + in * out; i++)
		p[i] = i;
	int32_t *bias = (int32_t*) (p + offset + in * out);
	for (int i = 0; i < out; i++)
		bias[i] = i + 1;
	LinearConfigQ lc =
	    { in, out, offset, scale, zero_point, out_scale, out_zero_point,
	    has_bias };
	uint8_t qx_out_q[batch_size * out];
	UQuantizedTensor qx_out =
	    { .q = qx_out_q, .scale = 0.0f, .zero_point = 0 };
	linear_q(&qx_out, &qx, p, lc);

	uint8_t ref[] = { 22, 2, 62, 202 };
	DOUBLES_EQUAL(lc.out_scale, qx_out.scale, TOL);
	DOUBLES_EQUAL(lc.out_zero_point, qx_out.zero_point, TOL);
	MEMCMP_EQUAL(ref, qx_out_q, batch_size * out * sizeof(uint8_t));
}

TEST(TestGroup, Im2col)
{
	int h = 2, w = 2, ksize = 2, stride = 2, pad = 1, ic = 2;
	uint8_t zp = 5;
	int col_size = ksize * ksize * ic * h * w;

	uint8_t im_q[] =
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	UQuantizedTensor im = { .q = im_q, .scale = 2.0f, .zero_point = zp };
	ConvConfigQ cc =
		{ ksize, stride, pad, ic, 0, 0, 0.0f, 0, 0.0f, 0, 0 }; // 0 not used
	uint8_t col_q[batch_size * col_size];
	UQuantizedTensor col = { .q = col_q, .scale = 0.0f, .zero_point = 0 };
	im2col_q(&col, &im, cc, &h, &w);

	uint8_t ref[] = {
		zp, zp, zp, 1, zp, zp, zp, 5, zp, zp, 2, zp, zp, zp, 6, zp,
		zp, 3, zp, zp, zp, 7, zp, zp, 4, zp, zp, zp, 8, zp, zp, zp,
		zp, zp, zp, 9, zp, zp, zp, 13, zp, zp, 10, zp, zp, zp, 14, zp,
		zp, 11, zp, zp, zp, 15, zp, zp, 12, zp, zp, zp, 16, zp, zp, zp
	};
	DOUBLES_EQUAL(im.scale, col.scale, TOL);
	DOUBLES_EQUAL(im.zero_point, col.zero_point, TOL);
	MEMCMP_EQUAL(ref, col.q, col_size * sizeof(uint8_t));
}

TEST(TestGroup, Conv)
{
	int h = 1, w = 1, ksize = 2, stride = 1, pad = 1, ic = 1, oc = 2;
	int offset = 1, has_bias = 1, zero_point = 1, out_zero_point = 2;
	float scale = 1.0f, out_scale = 3.0f;

	uint8_t qx_q[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	UQuantizedTensor qx = {.q = qx_q, .scale = 2.0f, .zero_point = 2};
	int w_size = ic * oc * ksize * ksize;
	int8_t p[offset + w_size + 4 * oc];
	for (int i = 0; i < offset + w_size; i++)
		p[i] = i;
	int32_t *bias = (int32_t*) (p + offset + w_size);
	for (int i = 0; i < oc; i++)
		bias[i] = i + 1;
	ConvConfigQ cc =
	    { ksize, stride, pad, ic, oc, offset, scale, zero_point, out_scale,
	    out_zero_point, has_bias };
	uint8_t qx_out_q[batch_size * oc * h * w];
	UQuantizedTensor qx_out =
	    {.q = qx_out_q, .scale = 0.0f, .zero_point = 0};
	conv_q(&qx_out, &qx, p, cc, h, w);

	uint8_t ref[] = { 8, 14, 24, 73 };
	DOUBLES_EQUAL(cc.out_scale, qx_out.scale, TOL);
	DOUBLES_EQUAL(cc.out_zero_point, qx_out.zero_point, TOL);
	MEMCMP_EQUAL(ref, qx_out.q, batch_size * oc * h * w * sizeof(uint8_t));
}

TEST(TestGroup, Maxpool)
{
	int nch = 2, h_before = 1, w_before = 1;
	int ksize = 2, stride = 1, pad = 1;
	int h_after = 2, w_after = 2;
	int out_size = nch * h_after * w_after;

	uint8_t qx_q[] = { 1, 2, 3, 4 };
	UQuantizedTensor qx = { .q = qx_q, .scale = 1.0f, .zero_point = 1 };
	uint8_t ref[] = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
	uint8_t qx_out_q[batch_size * out_size];
	UQuantizedTensor qx_out =
	    {.q = qx_out_q, .scale = 0.0f, .zero_point = 0};
	int h = h_before;
	int w = w_before;
	maxpool_q(&qx_out, &qx, &h, &w, nch, ksize, stride, pad);
	DOUBLES_EQUAL(h_after, h, TOL);
	DOUBLES_EQUAL(w_after, w, TOL);
	DOUBLES_EQUAL(qx.scale, qx_out.scale, TOL);
	DOUBLES_EQUAL(qx.zero_point, qx_out.zero_point, TOL);
	MEMCMP_EQUAL(ref, qx_out.q, batch_size * out_size * sizeof(uint8_t));
}

TEST(TestGroup, Relu)
{
	int size = 2;
	uint8_t qx_q[] = { 1, 2, 3, 4 };
	UQuantizedTensor qx = {.q = qx_q, .scale = 0.0f, .zero_point = 2};
	uint8_t ref[] = { 2, 2, 3, 4 };
	relu_q(&qx, size);
	MEMCMP_EQUAL(ref, qx.q, batch_size * size * sizeof(uint8_t));
}

TEST(TestGroup, Quantize_Dequantize)
{
	// quantize
	int size = 2, scale = 2.0f, zero_point = 1;
	float x[] = { 2.0f, 4.0f, 6.0f, 8.0f };
	uint8_t qx_q[batch_size * size];
	UQuantizedTensor qx = { .q = qx_q, .scale = 0.0f, .zero_point = 0 };
	quantize(&qx, x, scale, zero_point, size);
	uint8_t ref_q[] = { 2, 3, 4, 5 };
	MEMCMP_EQUAL(ref_q, qx.q, batch_size * size * sizeof(int8_t));
	DOUBLES_EQUAL(scale, qx.scale, TOL);
	DOUBLES_EQUAL(zero_point, qx.zero_point, TOL);

	// dequantize
	float xout[batch_size * size];
	dequantize(xout, &qx, size);
	float ref[] = { 2.0f, 4.0f, 6.0f, 8.0f };
	for (int i = 0; i < batch_size * size; i++) {
		DOUBLES_EQUAL(ref[i], xout[i], TOL);
	}
}

#endif
