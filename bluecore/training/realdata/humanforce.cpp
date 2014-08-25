#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
using namespace std;

#include "/usr/local/include/jpeglib.h"

typedef unsigned char BYTE;

const int INF = 19931005;
const int MAX_HEIGHT = 64;
const int MAX_WIDTH = 64;
const int FLOODFILL_DIST_X = 2;
const int FLOODFILL_DIST_Y = 2;
const int FLOODFILL_DIST = 2;
const int FLOODFILL_S_DIST = 1;
const int FLOODFILL_MIN_BLOCK = 10;
const int FLOODFILL_MIN_S_BLOCK = 4;
const int FLOODFILL_PREVENT_Y2_WIDTH = 6;
const int DIVIDER_CHAR_CNT = 4;
const int DIVIDER_MIN_WIDTH = 4;
const int ANN_HEIGHT_ROUGH = 24;
const int ANN_WIDTH_ROUGH = 24;
const int ANN_FRAC_X = 2;
const int ANN_FRAC_Y = 2;
const int ANN_HEIGHT = ANN_HEIGHT_ROUGH / ANN_FRAC_X;
const int ANN_WIDTH = ANN_WIDTH_ROUGH / ANN_FRAC_Y;
const int N_INPUT = ANN_HEIGHT * ANN_WIDTH, N_HIDDEN = 64, N_OUTPUT = 62;
BYTE bin[MAX_HEIGHT][MAX_WIDTH];
int h, w;
int global_id = 1169;

inline bool in (int x, int y) {
	return x >= 0 && x < h && y >= 0 && y < w;
}
inline BYTE get (int x, int y, BYTE data[][MAX_WIDTH] = bin) {
	if (in(x, y)) return data[x][y];
	else return 0;
}
inline BYTE median (int a0, int a1, int a2, int a3, int a4) {
	int arr[5] = {a0, a1, a2, a3, a4};
	sort(arr, arr + 5);
	return arr[2];
}
inline void median_filter (void) {
	BYTE tmp[128][128];
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			tmp[i][j] = median(get(i, j), get(i - 1, j), get(i, j - 1), get(i + 1, j), get(i, j + 1));
		}
	}
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			bin[i][j] = tmp[i][j];
		}
	}
}

int dx[128], dy[128], moves;
int s_dx[128], s_dy[128], s_moves;

bool sch[MAX_HEIGHT][MAX_WIDTH];
BYTE canvas[MAX_HEIGHT][MAX_WIDTH];
int x_min, x_max, y_min, y_max, flood_cnt;

void dfs (int x, int y, bool rec[][MAX_WIDTH], BYTE write[][MAX_WIDTH], BYTE data[][MAX_WIDTH], 
		int dx[], int dy[], int moves) {
	write[x][y] = true;
	++flood_cnt;
	x_min = min(x_min, x), x_max = max(x_max, x);
	y_min = min(y_min, y), y_max = max(y_max, y);
	for (int k = 0; k < moves; k++) {
		if (dy[k] >= 2 && y_max - y_min + 1 >= FLOODFILL_PREVENT_Y2_WIDTH) continue;
		int tx = x + dx[k], ty = y + dy[k];
		if (in(tx, ty) && !rec[tx][ty] && data[tx][ty]) {
			rec[tx][ty] = 1;
			dfs(tx, ty, rec, write, data, dx, dy, moves);
		}
	}
}
int floodfill (int x, int y) {
	x_min = y_min = INF, x_max = y_max = -INF;
	flood_cnt = 0;
	memset(canvas, 0, sizeof canvas);
	sch[x][y] = true;
	dfs(x, y, sch, canvas, bin, dx, dy, moves);

	if (flood_cnt < FLOODFILL_MIN_BLOCK) {
		return flood_cnt;
	}

	bool tsch[MAX_HEIGHT][MAX_WIDTH];
	BYTE tcanvas[MAX_HEIGHT][MAX_WIDTH];
	memset(tsch, 0, sizeof tsch);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (canvas[i][j] && !tsch[i][j]) {
				flood_cnt = 0;
				memset(tcanvas, 0, sizeof tcanvas);
				tsch[i][j] = true;
				dfs(i, j, tsch, tcanvas, canvas, s_dx, s_dy, s_moves);

				// Sweep
				if (flood_cnt < FLOODFILL_MIN_S_BLOCK) {
					for (int ti = 0; ti < h; ti++) {
						for (int tj = 0; tj < w; tj++) {
							if (tcanvas[ti][tj]) {
								canvas[ti][tj] = 0;
							}
						}
					}
				}
			}
		}
	}

	x_min = y_min = INF, x_max = y_max = -INF;
	flood_cnt = 0;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (canvas[i][j]) {
				++flood_cnt;
				x_min = min(x_min, i), x_max = max(x_max, i);
				y_min = min(y_min, j), y_max = max(y_max, j);
			}
		}
	}

	return flood_cnt;
}
BYTE ann_in_rough[ANN_HEIGHT_ROUGH][ANN_WIDTH_ROUGH];
char filename[256];
double ann_in[N_INPUT];
void ann_eval (int x_min, int x_max, int y_min, int y_max, BYTE data[][MAX_WIDTH] = canvas) {
	int x_off = (x_max - x_min) / 2 - ANN_HEIGHT_ROUGH / 2 + x_min;
	int y_off = (y_max - y_min) / 2 - ANN_WIDTH_ROUGH / 2 + y_min;

	memset(ann_in_rough, 0, sizeof ann_in_rough);
	for (int x = 0; x < ANN_HEIGHT_ROUGH; x++) if (x + x_off >= x_min && x + x_off <= x_max) {
		for (int y = 0; y < ANN_WIDTH_ROUGH; y++) if (y + y_off >= y_min && y + y_off <= y_max) {
			ann_in_rough[x][y] = get(x + x_off, y + y_off, data);
		}
	}

	printf("%s\n", filename);
	for (int x = 0; x < ANN_HEIGHT_ROUGH; x++) {
		for (int y = 0; y < ANN_WIDTH_ROUGH; y++) {
			printf("%c", ann_in_rough[x][y] ? '#' : '.');
		}
		printf("\n");
	}
	printf("\nANSWER: ");

	char tfile[256], human_ans[10];
	scanf("%s", human_ans);
	if (human_ans[0] != '/') {
		sprintf(tfile, "data%d.dat", global_id++);
		FILE *fp = fopen(tfile, "w");

		for (int x = 0; x < ANN_HEIGHT; x++) {
			for (int y = 0; y < ANN_WIDTH; y++) {
				int pl = x * ANN_WIDTH + y;
				ann_in[pl] = 0.0;
				for (int xx = 0; xx < ANN_FRAC_X; xx++) {
					for (int yy = 0; yy < ANN_FRAC_Y; yy++) {
						ann_in[pl] += ann_in_rough[x * 2 + xx][y * 2 + yy];
					}
				}
			}
		}

		for (int i = 0; i < N_INPUT; i++) {
			fprintf(fp, "%.0lf ", ann_in[i]);
		}
		fprintf(fp, "\n");
		fprintf(fp, "%c\n", human_ans[0]);
		fclose(fp);
	}
}
struct letter_box {
	int x_min, x_max, y_min, y_max;
	BYTE l_canvas[MAX_HEIGHT][MAX_WIDTH];

	letter_box (void) {}
	letter_box (int _xmin, int _xmax, int _ymin, int _ymax, BYTE _canvas[][MAX_WIDTH]) : x_min(_xmin), x_max(_xmax), y_min(_ymin), y_max(_ymax) {
		for (int i = 0; i < MAX_HEIGHT; i++) {
			for (int j = 0; j < MAX_WIDTH; j++) {
				l_canvas[i][j] = _canvas[i][j];
			}
		}
	}

	void modify_box (int _xmin, int _xmax, int _ymin, int _ymax) {
		x_min = _xmin, x_max = _xmax, y_min = _ymin, y_max = _ymax;
	}

	void add_canvas (BYTE _canvas[][MAX_WIDTH]) {
		for (int i = 0; i < MAX_HEIGHT; i++) {
			for (int j = 0; j < MAX_WIDTH; j++) {
				if (_canvas[i][j]) l_canvas[i][j] = 1;
			}
		}
	}

	int width (void) {
		return y_max - y_min + 1;
	}

	void eval (void) {
		ann_eval(x_min, x_max, y_min, y_max, l_canvas);
	}
};
letter_box candidate[10]; int c_max;

void divide (int idx) {
	for (int i = c_max - 1; i >= idx + 1; i--) {
		candidate[i + 1] = candidate[i];
	}
	++c_max;
	letter_box t = candidate[idx];
	int len = t.width(), cut = -1, min_char = INF;
	for (int j = t.y_min + DIVIDER_MIN_WIDTH - 1; j <= t.y_max - DIVIDER_MIN_WIDTH; j++) {
		int c_char = 0;
		for (int i = t.x_min; i <= t.x_max; i++) {
			c_char += t.l_canvas[i][j];
		}
		if (c_char < min_char) {
			min_char = c_char;
			cut = j;
		}
	}
	candidate[idx].modify_box(t.x_min, t.x_max, t.y_min, cut);
	candidate[idx + 1] = letter_box(t.x_min, t.x_max, cut + 1, t.y_max, t.l_canvas);
}
void merge (int idx) {
	letter_box t1 = candidate[idx], t2 = candidate[idx + 1];
	candidate[idx].modify_box(min(t1.x_min, t2.x_min), max(t1.x_max, t2.x_max), min(t1.y_min, t2.y_min), max(t1.y_max, t2.y_max));
	candidate[idx].add_canvas(t2.l_canvas);
	for (int i = idx + 1; i < c_max - 1; i++) {
		candidate[i] = candidate[i + 1];
	}
	--c_max;
}

void work (char *filename) {
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);

	FILE *fp = fopen(filename, "rb");

	jpeg_stdio_src(&cinfo, fp);

	jpeg_read_header(&cinfo, TRUE);

	BYTE* data = new BYTE[cinfo.image_width * cinfo.image_height * cinfo.num_components];
	w = cinfo.image_width, h = cinfo.image_height;
	int cl = cinfo.num_components;

	jpeg_start_decompress(&cinfo);

	JSAMPROW row;
	while (cinfo.output_scanline < cinfo.output_height) {
		row = &data[(cinfo.output_height - cinfo.output_scanline - 1) * cinfo.image_width * cinfo.num_components];
		jpeg_read_scanlines(&cinfo, &row, 1);
	}
	jpeg_finish_decompress(&cinfo);

	jpeg_destroy_decompress(&cinfo);
	fclose(fp);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int pt = cl * ((h - 1 - i) * w + j);
			bin[i][j] = (data[pt] > 128 ? 0 : 1);
		}
	}

	// Median Filter
	median_filter();

	// Preproceed Moves
	moves = 0;
	for (int i = -FLOODFILL_DIST_X; i <= FLOODFILL_DIST_X; i++) {
		for (int j = -FLOODFILL_DIST_Y; j <= FLOODFILL_DIST_Y; j++) {
			if (abs(i) + abs(j) <= FLOODFILL_DIST) {
				dx[moves] = i, dy[moves] = j; ++moves;
			}
		}
	}
	s_moves = 0;
	for (int i = -FLOODFILL_S_DIST; i <= FLOODFILL_S_DIST; i++) {
		for (int j = -FLOODFILL_S_DIST; j <= FLOODFILL_S_DIST; j++) {
			if (abs(i) + abs(j) <= FLOODFILL_S_DIST) {
				s_dx[s_moves] = i, s_dy[s_moves] = j; ++s_moves;
			}
		}
	}

	// Initialize
	c_max = 0;

	// Floodfill
	memset(sch, 0, sizeof sch);
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < h; i++) {
			if (!sch[i][j] && bin[i][j]) {
				int cnt = floodfill(i, j);
				if (cnt >= FLOODFILL_MIN_BLOCK) {
					candidate[c_max++] = letter_box(x_min, x_max, y_min, y_max, canvas);
				}
			}
		}
	}

	// Divide
	while (c_max < DIVIDER_CHAR_CNT) {
		int m = 0;
		for (int i = 1; i < c_max; i++) if (candidate[i].width() > candidate[m].width()) m = i;
		divide(m);
	}
	// Merge
	while (c_max > DIVIDER_CHAR_CNT) {
		int m = 0;
		for (int i = 1; i + 1 < c_max; i++)
			if (candidate[i].width() + candidate[i + 1].width() < candidate[m].width() + candidate[m + 1].width()) 
				m = i;
		merge(m);
	}

	// Eval
	for (int i = 0; i < c_max; i++) {
		candidate[i].eval();
	}
}

int main (int argc, char *argv[]) {
	int s = atoi(argv[1]), t = atoi(argv[2]);
	for (int i = s; i <= t; i++) {
		sprintf(filename, "cap%d.jpg", i);
		work(filename);
	}
	return 0;
}
