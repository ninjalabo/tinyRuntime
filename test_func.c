#include "CppUTest/TestHarness.h"
#include "func.h"
TEST_GROUP(Softmax)
{
	void setup() {
	}
	void teardown() {
	}
};


TEST(Softmax, OutputSumsToOne)
{
	float x[] = { 1.0, 2.0, 3.0, };
	softmax(x, 3);
	DOUBLES_EQUAL(x[0]+x[1]+x[2], 1.0, 0.01);
}
