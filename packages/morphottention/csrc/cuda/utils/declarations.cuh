#ifndef MORPHOTTENTION_DECLARATIONS_CUH
#define MORPHOTTENTION_DECLARATIONS_CUH


// ---------------------------------------------------------------------------
// Threadblock geometry
// ---------------------------------------------------------------------------
#ifndef KERNEL_BLOCK_SIZE
#define KERNEL_BLOCK_SIZE 128
#endif

constexpr int BLOCK_SIZE = KERNEL_BLOCK_SIZE;
constexpr int WARPS      = BLOCK_SIZE / 32;

// ---------------------------------------------------------------------------
// Sequence tiles (forward)
// ---------------------------------------------------------------------------
#ifndef KERNEL_BR_FWD
#define KERNEL_BR_FWD 64
#endif

#ifndef KERNEL_BC_FWD
#define KERNEL_BC_FWD 64
#endif

constexpr int BR_FWD = KERNEL_BR_FWD;
constexpr int BC_FWD = KERNEL_BC_FWD;

// ---------------------------------------------------------------------------
// Feature dims
// ---------------------------------------------------------------------------
#ifndef KERNEL_CUBE_M
#define KERNEL_CUBE_M 64
#endif

#ifndef KERNEL_HEAD_DIM_V
#define KERNEL_HEAD_DIM_V 64
#endif

constexpr int CUBE_M_FWD     = KERNEL_CUBE_M;
constexpr int HEAD_DIM_V_FWD = KERNEL_HEAD_DIM_V;

constexpr int TILE_M_FWD = BR_FWD;
constexpr int TILE_N_FWD = BC_FWD;

// ---------------------------------------------------------------------------
// Backward tiles
// ---------------------------------------------------------------------------
#ifndef KERNEL_BR_BWD
#define KERNEL_BR_BWD 64
#endif

#ifndef KERNEL_BC_BWD
#define KERNEL_BC_BWD 64
#endif

constexpr int BR_BWD = KERNEL_BR_BWD;
constexpr int BC_BWD = KERNEL_BC_BWD;
constexpr int TILE_M_BWD = BR_BWD;
constexpr int TILE_N_BWD = BC_BWD;

// ---------------------------------------------------------------------------
// Online-softmax flags
// ---------------------------------------------------------------------------
#ifndef MORPHO_CAUSAL
#define MORPHO_CAUSAL 0
#endif
constexpr bool CAUSAL = (MORPHO_CAUSAL != 0);


#ifndef MORPHO_MASK_DIAG
#define MORPHO_MASK_DIAG 0
#endif
constexpr bool MASK_DIAG = (MORPHO_MASK_DIAG != 0);

static_assert(BR_FWD % 16 == 0, "BR_FWD must be %16 (MMA M)");
static_assert(BC_FWD % 16 == 0, "BC_FWD must be %16 (it is K of GEMM-2)");
static_assert(CUBE_M_FWD % 16 == 0, "CUBE_M must be %16 (K of GEMM-1)");
static_assert(HEAD_DIM_V_FWD % 8 == 0, "HEAD_DIM_V must be %8 (N of GEMM-2)");
static_assert(BR_BWD % 16 == 0 && BC_BWD % 16 == 0, "bwd tiles must align to MMA atom");
static_assert((BR_FWD * BC_FWD) % (WARPS * 16 * 8) == 0,"score tile does not partition evenly across warps");


#endif // MORPHOTTENTION_DECLARATIONS_CUH