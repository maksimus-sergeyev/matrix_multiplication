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
				C.randomfill();
				D.randomfill();

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
				C.randomfill();
				D.randomfill();

				C = A * B;

				parallel_mult(A, B, D);

				if ((D - C).norm() > EPS) return false;

			}

	return true;

}

bool block_mult_correctness(double EPS)
{
	const int sz = 64;
	const int bsz = 32;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++)
				for (int i = 1; i < 10; i++)
					for (int j = 1; j < 10; j++)
					{

						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

						A.randomfill();
						B.randomfill();
						C.randomfill();
						D.randomfill();

						C = A * B;

						block_mult(A, B, D, i, j);

						if ((D - C).norm() > EPS) return false;

					}

	for (int row1 = 8; row1 <= sz; row1 += 8)
		for (int col1 = 8; col1 <= sz; col1 += 8)
			for (int col2 = 8; col2 <= sz; col2 += 8)

				for (int i = 8; i < bsz; i += 8)
					for (int j = 8; j < bsz; j += 8)
					{

						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

						A.randomfill();
						B.randomfill();
						C.randomfill();
						D.randomfill();

						C = A * B;

						block_mult(A, B, D, i, j);

						if ((D - C).norm() > EPS) return false;

					}

	return true;

}

bool parallel_block_mult_correctness(double EPS)
{
	const int sz = 64;
	const int bsz = 32;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++) 
				for (int i = 1; i < 10; i++)
					for (int j = 1; j < 10; j++)
					{

						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

						A.randomfill();
						B.randomfill();
						C.randomfill();
						D.randomfill();

						C = A * B;

						parallel_block_mult(A, B, D, i, j);

						if ((D - C).norm() > EPS) return false;

					}

	for (int row1 = 8; row1 <= sz; row1 += 8)
		for (int col1 = 8; col1 <= sz; col1 += 8)
			for (int col2 = 8; col2 <= sz; col2 += 8)

				for (int i = 8; i < bsz; i += 8)
					for (int j = 8; j < bsz; j += 8)
					{

						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

						A.randomfill();
						B.randomfill();
						C.randomfill();
						D.randomfill();

						C = A * B;

						parallel_block_mult(A, B, D, i, j);

						if ((D - C).norm() > EPS) return false;

					}

	return true;

}

bool parallel_block_mult2_correctness(double EPS)
{
	const int sz = 64;
	const int bsz = 32;

	srand(time(NULL));

	for (int row1 = 1; row1 <= 10; row1++)
		for (int col1 = 1; col1 <= 10; col1++)
			for (int col2 = 1; col2 <= 10; col2++)
				for (int i = 1; i < 10; i++)
					for (int j = 1; j < 10; j++)
					{

						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

						A.randomfill();
						B.randomfill();
						C.randomfill();
						D.randomfill();

						C = A * B;

						parallel_block_mult2(A, B, D, i, j);

						if ((D - C).norm() > EPS) return false;

					}

	for (int row1 = 8; row1 <= sz; row1 += 8)
		for (int col1 = 8; col1 <= sz; col1 += 8)
			for (int col2 = 8; col2 <= sz; col2 += 8)

				for (int i = 8; i < bsz; i += 8)
					for (int j = 8; j < bsz; j += 8)
					{

						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

						A.randomfill();
						B.randomfill();
						C.randomfill();
						D.randomfill();

						C = A * B;

						parallel_block_mult2(A, B, D, i, j);

						if ((D - C).norm() > EPS) return false;

					}

	return true;

}

bool parallel_block_mult3_correctness(double EPS)
{
	const int sz = 32;
	const int bsz = 16;

	srand(time(NULL));

	for (int row1 = 4; row1 <= sz; row1 += 4)
		for (int col1 = 4; col1 <= sz; col1 += 4)
			for (int col2 = 4; col2 <= sz; col2 += 4)
				for (int i = 4; (i <= bsz) && (i<row1) && (i < col1) && (i < col2); i += 4)
					for (int j = 4; (j <= bsz)&& (j < sz) && (j< row1) && (j < col1) && (j < col2); j += 4)
						for (int k = 4; (k <= j)&& (k <= i); k += 4)
							if ((j % k == 0) && (i % k == 0)) {

								int row2 = col1;

								matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2);

								A.randomfill();
								B.randomfill();
								C.randomfill();
								D.randomfill();

								C = A * B;

								parallel_block_mult3(A, B, D, i, j, k);

								if ((D - C).norm() > EPS) return false;
							}

	return true;

}
