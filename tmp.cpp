#pragma once
#include "matrix.h"
#include <time.h>
#include <chrono>
#include <iomanip>

bool mult_correctness(double EPS) 
{
	const int sz = 32;

	srand(time(NULL));

	for (int row1 = 1; row1 <= sz; row1 ++)
		for (int col1 = 1; col1 <= sz; col1 ++)
			for (int col2 = 1; col2 <= sz; col2 ++)
			{
				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				C = A * B;

				mult(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;
}

bool parallel_mult_correctness(double EPS)
{
	const int sz = 32;

	srand(time(NULL));

	for (int row1 = 1; row1 <= sz; row1++)
		for (int col1 = 1; col1 <= sz; col1++)
			for (int col2 = 1; col2 <= sz; col2++)
			{
				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				C = A * B;

				parallel_mult(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;

}

bool block_mult_correctness(double EPS)
{
	const int sz = 1024;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++)
					{
						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

						A.randomfill();
						B.randomfill();

						parallel_mult(A, B, C);

						block_mult(A, B, D);

						if ((D - C).norm() > EPS) return false;

					}

	for (int row1 = 256; row1 <= sz; row1 *=2 )
		for (int col1 = 256; col1 <= sz; col1 *= 2)
			for (int col2 = 256; col2 <= sz; col2 *= 2)
					{

						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

						A.randomfill();
						B.randomfill();

						parallel_mult(A, B, C);

						block_mult(A, B, D);

						if ((D - C).norm() > EPS) return false;

					}

	for (int row1 = 200; row1 <= sz; row1 *= 2)
		for (int col1 = 200; col1 <= sz; col1 *= 2)
			for (int col2 = 200; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				block_mult(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;

}

bool parallel_block_mult_correctness(double EPS)
{
	const int sz = 1024;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++)
			{
				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 256; row1 <= sz; row1 *= 2)
		for (int col1 = 256; col1 <= sz; col1 *= 2)
			for (int col2 = 256; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 200; row1 <= sz; row1 *= 2)
		for (int col1 = 200; col1 <= sz; col1 *= 2)
			for (int col2 = 200; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;

}

bool parallel_block_mult2_correctness(double EPS)
{
	const int sz = 1024;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++)
			{
				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult2(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 256; row1 <= sz; row1 *= 2)
		for (int col1 = 256; col1 <= sz; col1 *= 2)
			for (int col2 = 256; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult2(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 200; row1 <= sz; row1 *= 2)
		for (int col1 = 200; col1 <= sz; col1 *= 2)
			for (int col2 = 200; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult2(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;

}

bool parallel_block_mult3_correctness(double EPS)
{
	const int sz = 1024;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++)
			{
				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult3(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 256; row1 <= sz; row1 *= 2)
		for (int col1 = 256; col1 <= sz; col1 *= 2)
			for (int col2 = 256; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult3(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 200; row1 <= sz; row1 *= 2)
		for (int col1 = 200; col1 <= sz; col1 *= 2)
			for (int col2 = 200; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult3(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;

}

bool parallel_block_mult4_correctness(double EPS)
{
	const int sz = 1024;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++)
			{
				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult4(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 256; row1 <= sz; row1 *= 2)
		for (int col1 = 256; col1 <= sz; col1 *= 2)
			for (int col2 = 256; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult4(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 200; row1 <= sz; row1 *= 2)
		for (int col1 = 200; col1 <= sz; col1 *= 2)
			for (int col2 = 200; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult4(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;

}

bool parallel_block_mult5_correctness(double EPS)
{
	const int sz = 1024;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++)
			{
				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult5(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 256; row1 <= sz; row1 *= 2)
		for (int col1 = 256; col1 <= sz; col1 *= 2)
			for (int col2 = 256; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult5(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	for (int row1 = 200; row1 <= sz; row1 *= 2)
		for (int col1 = 200; col1 <= sz; col1 *= 2)
			for (int col2 = 200; col2 <= sz; col2 *= 2)
			{

				int row2 = col1;

				matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

				A.randomfill();
				B.randomfill();

				parallel_mult(A, B, C);

				parallel_block_mult5(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;

}
