#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
using namespace std;

const int H = 12, W = 12;
const int RH = 24, RW = 24;
const int N_INPUT = H * W, N_HIDDEN = 64, N_OUTPUT = 62;
const int CNT_SAMPLE = 30;
const int MAX_OFFSET = 1;
const int CNT_TRAINDATA = CNT_SAMPLE * N_OUTPUT * (MAX_OFFSET * 2 + 1) * (MAX_OFFSET * 2 + 1);
const double ALPHA = 0.2;
const double ALLOW_ERROR = 1e-04;
const int MAX_ITERATE = 2000000;

double w_ih[N_INPUT + 1][N_HIDDEN], w_ho[N_HIDDEN + 1][N_OUTPUT];

double phi (double z) {
	return 1 / (1 + exp(-z));
}

double in_h[N_HIDDEN], out_h[N_HIDDEN], in_o[N_OUTPUT], out_o[N_OUTPUT];

inline int code (char ch) {
	if (ch <= '9') return ch - '0';
	else if (ch <= 'Z') return ch - 'A' + 10;
	else return ch - 'a' + 36;
}
inline char decode (int c) {
	if (c < 10) return c + '0';
	else if (c < 36) return c - 10 + 'A';
	else return c - 36 + 'a';
}

inline double random_float (void) {
	return (rand() % 20000 - 10000) * 0.0001;
}

void net_calculate (double *in) {
	// Input -> Hidden In
	for (int i = 0; i < N_HIDDEN; i++) {
		in_h[i] = 0.0;
		for (int j = 0; j < N_INPUT; j++) {
			in_h[i] += in[j] * w_ih[j][i];
		}
		in_h[i] -= w_ih[N_INPUT][i];
	}

	// Hidden In -> Hidden Out
	for (int i = 0; i < N_HIDDEN; i++) {
		out_h[i] = phi(in_h[i]);
	}

	// Hidden Out -> Output In
	for (int i = 0; i < N_OUTPUT; i++) {
		in_o[i] = 0.0;
		for (int j = 0; j < N_HIDDEN; j++) {
			in_o[i] += out_h[j] * w_ho[j][i];
		}
		in_o[i] -= w_ho[N_HIDDEN][i];
	}

	// Output In -> Output Out
	for (int i = 0; i < N_OUTPUT; i++) {
		out_o[i] = phi(in_o[i]);
	}
}

double delta_h[N_HIDDEN], delta_o[N_OUTPUT];

double train_in[CNT_TRAINDATA][N_INPUT];
double train_out[CNT_TRAINDATA][N_OUTPUT];

int main () {
	// Read training data
	int t_id = 0;
	for (int i = 0; i < CNT_SAMPLE; i++) {
		for (int j = 0; j < N_OUTPUT; j++) if (decode(j) != '0' && decode(j) != 'o' && decode(j) != 'O' && decode(j) != 'l' && decode(j) != 'I') {
			char tfile[256];
			sprintf(tfile, "data/%02d-%d", i, decode(j));
			char row[RH][RW + 1];
			FILE *fp = fopen(tfile, "r");
			for (int x = 0; x < RH; x++) {
				fscanf(fp, "%s", row[x]);
			}
			fclose(fp);
			for (int dx = -MAX_OFFSET; dx <= MAX_OFFSET; ++dx) {
				for (int dy = -MAX_OFFSET; dy <= MAX_OFFSET; ++dy) {
					memset(train_in[t_id], 0, sizeof train_in[0]);
					for (int x = 0; x < H; x++) {
						for (int y = 0; y < W; y++) {
							int pl = x * W + y;
							for (int xx = 0; xx < 2; ++xx) if (x * 2 + xx + dx >= 0 && x * 2 + xx + dx < RH) {
								for (int yy = 0; yy < 2; ++yy) if (y * 2 + yy + dy >= 0 && y * 2 + yy + dy < RW) {
									if (row[x * 2 + xx + dx][y * 2 + yy + dy] == '#') {
										train_in[t_id][pl] += 1.0;
									}
								}
							}
						}
					}

					/* printf("DATA: %d [\n", t_id);
					for (int k = 0; k < N_INPUT; k++) {
						printf(" %.0f", train_in[t_id][k]);
						if (k % W == W - 1) printf("\n");
					}
					printf(" ]\n"); */

					for (int k = 0; k < N_OUTPUT; k++) {
						train_out[t_id][k] = (k == j ? 1.0 : 0.0);
					}
					++t_id;
				}
			}
		}
	}

	printf("READ ALL DATA. T_ID = %d\n", t_id);
	// getchar();

	// Randomize weights
	srand(3939);
	FILE *fsave = fopen("weights.log", "r");
	for (int i = 0; i < N_INPUT; i++) {
		for (int j = 0; j < N_HIDDEN; j++) {
			// w_ih[i][j] = random_float();
			fscanf(fsave, "%lf", &w_ih[i][j]);
		}
	}
	for (int i = 0; i < N_HIDDEN; i++) {
		for (int j = 0; j < N_OUTPUT; j++) {
			// w_ho[i][j] = random_float();
			fscanf(fsave, "%lf", &w_ho[i][j]);
		}
	}

	// Train
	int trained = 0, train = 0;
	while (1) {
		// Randomize a data to train
		// train = rand() % t_id;

		// Calculate in net
		net_calculate(train_in[train]);

		// Propagate: Calculate delta_o
		for (int i = 0; i < N_OUTPUT; i++) {
			delta_o[i] = (out_o[i] - train_out[train][i]) * out_o[i] * (1 - out_o[i]);
		}

		// Propagate: Calculate delta_h
		for (int i = 0; i < N_HIDDEN; i++) {
			delta_h[i] = 0.0;
			for (int j = 0; j < N_OUTPUT; j++) {
				delta_h[i] += delta_o[j] * w_ho[i][j];
			}
			delta_h[i] *= out_h[i] * (1 - out_h[i]);
		}

		// Gradient descent
		for (int j = 0; j < N_HIDDEN; j++) {
			for (int i = 0; i < N_INPUT; i++) {
				w_ih[i][j] -= ALPHA * delta_h[j] * train_in[train][i];
			}
			w_ih[N_INPUT][j] += ALPHA * delta_h[j];
		}
		for (int j = 0; j < N_OUTPUT; j++) {
			for (int i = 0; i < N_HIDDEN; i++) {
				w_ho[i][j] -= ALPHA * delta_o[j] * out_h[i];
			}
			w_ho[N_HIDDEN][j] += ALPHA * delta_o[j];
		}

		if (trained % 100000 == 0) {
			// Calculate global error
			double error = 0.0, max_error = 0.0; train = 0;
			int wrong = 0;
			for (int k = 0; k < t_id; k++) {
				net_calculate(train_in[k]);
				int t_ans = 0, o_ans = 0;
				double te = 0;
				for (int o = 0; o < N_OUTPUT; o++) {
					double d = (out_o[o] - train_out[k][o]) * (out_o[o] - train_out[k][o]);
					te += d;
					error += d;
					if (out_o[o] > out_o[o_ans]) o_ans = o;
					if (train_out[k][o] > train_out[k][t_ans]) t_ans = o;
				}
				if (0.5 * te > max_error) {
					max_error = 0.5 * te;
					train = k;
				}
				if (o_ans != t_ans)
				{
					++wrong;
					/* printf("DATA %d: O_ANS = %c (%lf), T_ANS = %c (%lf)\n", k, decode(o_ans), out_o[o_ans], decode(t_ans), train_out[k][t_ans]);
					for (int o = 0; o < N_OUTPUT; o++) {
						printf("%c: %lf (%lf)\n", decode(o), out_o[o], in_o[o]);
					}
					getchar(); */
				}
			}
			error /= (2 * t_id);
			printf("TRAINED %d : ERROR %lf, MAX_ERROR = %lf, WRONG = %d (%.2lf%%)\n", trained, error, max_error, wrong, (double)wrong * 100 / t_id);
			if (fabs(error) < ALLOW_ERROR || trained >= MAX_ITERATE) {
				break;
			}
		}
		else {
			train = rand() % t_id;
		}

		++trained;
	}

	FILE *fout = fopen("weights.log", "w");
	for (int i = 0; i < N_INPUT; i++) {
		for (int j = 0; j < N_HIDDEN; j++) {
			fprintf(fout, "%.8lf ", w_ih[i][j]);
		}
		fprintf(fout, "\n");
	}
	fprintf(fout, "\n");
	for (int i = 0; i < N_HIDDEN; i++) {
		for (int j = 0; j < N_OUTPUT; j++) {
			fprintf(fout, "%.8lf ", w_ho[i][j]);
		}
		fprintf(fout, "\n");
	}
	fprintf(fout, "\n");
	fclose(fout);

	return 0;
}
