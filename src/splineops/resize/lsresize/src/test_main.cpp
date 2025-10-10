// splineops/src/splineops/resize/lsresize/src/test_main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <numeric>
#include "lsresize/resize1d.h"
#include "lsresize/resizend.h"

using namespace lsresize;

// Simple helpers
static int64_t prod(const std::vector<int64_t>& shape) {
  int64_t p = 1; for (auto x : shape) p *= x; return p;
}

static std::vector<int64_t> shape_of_2d(int h, int w) { return {h, w}; }

// N-D resize wrapper (like the pybind binding, but pure C++)
static std::vector<double> resize_nd(const std::vector<double>& data,
                                     const std::vector<int64_t>& in_shape,
                                     const std::vector<double>& zoom,
                                     const std::string& algo,
                                     int degree,
                                     bool inversable=false)
{
  const int D = (int)in_shape.size();
  std::vector<int64_t> out_shape = in_shape;
  for (int i=0;i<D;++i) out_shape[i] = (int64_t)std::llround((double)in_shape[i] * zoom[i]);

  std::vector<double> tmp_in = data;
  std::vector<int64_t> cur_shape = in_shape;

  for (int ax = 0; ax < D; ++ax) {
    std::vector<int64_t> next_shape = cur_shape;
    next_shape[ax] = out_shape[ax];
    std::vector<double> tmp_out((size_t)prod(next_shape), 0.0);

    int interp_degree = degree;
    int synthe_degree = degree;
    int analy_degree  = degree;
    if (algo == "interpolation") analy_degree = -1;
    else if (algo == "oblique") analy_degree = (degree == 1 ? 0 : 1);

    LSParams p{interp_degree, analy_degree, synthe_degree, zoom[ax], 0.0, inversable};
    resize_along_axis(tmp_in.data(), tmp_out.data(), cur_shape, next_shape, ax, p);

    tmp_in.swap(tmp_out);
    cur_shape.swap(next_shape);
  }
  return tmp_in;
}

static double mse(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) { std::cerr << "Size mismatch\n"; return 1e9; }
  double acc = 0.0; for (size_t i=0;i<a.size();++i) { double d = a[i]-b[i]; acc += d*d; }
  return acc / (double)a.size();
}

// Flatten 2D initializer into row-major vector
template<int H, int W>
static std::vector<double> flat(const double (&M)[H][W]) {
  std::vector<double> v; v.reserve(H*W);
  for (int i=0;i<H;++i) for (int j=0;j<W;++j) v.push_back(M[i][j]);
  return v;
}

int main() {
  // --- Build the 10x10 square (like your Python test) ---
  // zeros with a 4x4 block of ones at [3:7, 3:7], normalized to 1.0
  const int H = 10, W = 10;
  std::vector<double> img(H*W, 0.0);
  for (int y = 3; y < 7; ++y)
    for (int x = 3; x < 7; ++x)
      img[y*W + x] = 1.0;

  // Reference matrices from your Python test (cubic least-squares)
  static const double DOWNSCALED_0_5_JAVA_SQUARE_LS_3_3_3[5][5] = {
    { 0.0018689470814154346, -0.008251404860385576, -0.04985014714990277, -0.03066989849060565, 0.0013938326549015785},
    {-0.008251404860385763,  0.036429967893169046,  0.22008849291341548,  0.1354076592052344, -0.006153773778607437},
    {-0.04985014714990389,   0.2200884929134171,    1.3296455504694955,   0.8180536346012451, -0.03717749081293505},
    {-0.030669898490606213,  0.13540765920523523,   0.8180536346012426,   0.5033008600284541, -0.022873149520289068},
    { 0.0013938326549015941, -0.006153773778607329, -0.037177490812933506,-0.022873149520288183,0.0010394994535632193}
  };

  static const double REVERTED_0_5_JAVA_SQUARE_LS_3_3_3[10][10] = {
    { 0.0018781323400760897,  0.0010571984329069463, -0.008262973560162083, -0.031004085484624598, -0.04994430169807815, -0.048221066443239174, -0.03064737786618124, -0.00944189116442014,  0.0017383923557298085, -0.009896553700642022},
    { 0.0010571984329074248,  5.950957249894561E-4, -0.004651217868177059, -0.01745216238958293,  -0.028113587291633116, -0.02714358024147593,  -0.017251372100620554, -0.005314829168172281, 9.785389639727809E-4,  -0.005570758162376665},
    {-0.008262973560158452,  -0.004651217868181094,  0.03635352557381595,   0.13640462556876864,   0.21973342112605143,   0.2121519280410835,    0.13483526564821757,   0.04154025538284788,  -0.007648177801984145,  0.04354057476150174},
    {-0.031004085484616906,  -0.017452162389586988,  0.13640462556877103,   0.5118134096450134,    0.8244772566221064,     0.7960302020605083,     0.5059249036758746,     0.15586612005549,     -0.028697266988463928,  0.1633716593823823},
    {-0.04994430169808113,   -0.02811358729163959,   0.21973342112606764,   0.8244772566220873,    1.3281456364312492,     1.2823204410339848,     0.8149914964688332,     0.2510838298527845,    -0.04622826114622253,   0.2631744596742541},
    {-0.04822106644322524,   -0.027143580241484618,  0.21215192804110306,   0.7960302020604646,    1.2823204410340732,     1.238076358788619,       0.7868717304218792,     0.24242064921323506,   -0.044633240960376956,  0.25409411433617746},
    {-0.030647377866186362,  -0.017251372100625433,  0.1348352656482193,    0.5059249036758733,    0.8149914964688023,      0.7868717304218127,      0.5001041460341442,      0.15407285211640753,   -0.02836709973464651,   0.16149203885400695},
    {-0.00944189116442196,   -0.005314829168163082,  0.04154025538285008,   0.15586612005550343,   0.2510838298528118,      0.24242064921333514,     0.15407285211635763,     0.04746700051885365,   -0.008739379581333187,  0.049752714944766116},
    { 0.001738392355726042,   9.785389639690011E-4, -0.007648177801995245, -0.02869726698848693,  -0.04622826114624887,    -0.0446332409604733,     -0.028367099734603184,   -0.008739379581321288,  0.001609049542452153,  -0.00916021354522421},
    {-0.009896553700638518,  -0.005570758162372193,  0.0435405747615197,    0.1633716593823974,    0.26317445967427133,     0.2540941143362428,      0.16149203885396476,     0.049752714944765124,  -0.00916021354522903,   0.05214849510856498}
  };

  // --- Run C++ resize: down (0.5, 0.5) then up to (10, 10) ---
  std::vector<int64_t> in_shape = shape_of_2d(H, W);
  std::vector<double> zoom_down = {0.5, 0.5};
  std::vector<double> zoom_up   = {2.0, 2.0};

  auto down = resize_nd(img, in_shape, zoom_down, "least-squares", /*degree=*/3);
  auto up   = resize_nd(down, {5,5},  zoom_up,   "least-squares", /*degree=*/3);

  auto ref_down = flat(DOWNSCALED_0_5_JAVA_SQUARE_LS_3_3_3);
  auto ref_up   = flat(REVERTED_0_5_JAVA_SQUARE_LS_3_3_3);

  double mse_down = mse(down, ref_down);
  double mse_up   = mse(up,   ref_up);

  std::cout << "MSE down (LS cubic, 0.5x): " << mse_down << "\n";
  std::cout << "MSE up   (revert to 10x10): " << mse_up   << "\n";

  const double tol_down = 1e-3;
  const double tol_up   = 1e-3;

  if (mse_down < tol_down && mse_up < tol_up) {
    std::cout << "OK : both tests passed.\n";
    return 0;
  }
  std::cerr << "FAIL: tolerances exceeded.\n";
  return 1;
}
