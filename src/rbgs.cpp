// ===========================================================================
// Step 2: RBGS (Red-Black Gauss-Seidel) 平滑器
// 编译: g++ -std=c++17 -O2 rbgs.cpp -o rbgs
// 运行: ./rbgs [N]
// ===========================================================================
//
// 【为什么 Jacobi 不够用】
// Jacobi 每轮用旧值同步更新所有点 → 信息传播速度 = 1 格/轮
// N=64 的网格, 中心点的误差要传 32 轮才到边界 → 需上万轮
//
// 【Gauss-Seidel: 算完立刻用新值】
// 逐点更新, 第 (i,j) 点更新时, (i-1,j) 和 (i,j-1) 已经是新值
// 信息传播更快 → 收敛约快一倍
// 但问题是: 有顺序依赖, 无法并行
//
// 【RBGS: 红黑染色解决并行问题】
// 5 点 Laplacian 模板下, 每个点只连接上下左右
// 棋盘染色: (i+j) 偶=红, 奇=黑
//   → 红点只连黑点, 黑点只连红点
//   → 所有红点之间互不相邻 → 可同时更新!
//   → 红点更新完后, 所有黑点也可同时更新!
//
// 效果: 收敛接近 Gauss-Seidel, 但每色内部完全并行
// 这就是论文 Algorithm 3 (V-Cycle) 里每一层的平滑器
//
// ===========================================================================
// 【RBGS 更新公式】
// ===========================================================================
// 与 Jacobi 相同: x_{i,j} += D⁻¹ (b_{i,j} - (Ax)_{i,j})
//                 x_{i,j} += (h²/4) * [b_{i,j} - (4x_{ij}-Σ邻居)/h²]
//
// 区别: Jacobi 的 Ax 全读旧值, RBGS 的 Ax 读当前最新值(混合新旧)
//       Jacobi 需要 x_new 备份数组, RBGS 原地修改
// ===========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

struct Grid {
    int N;
    double h;
    std::vector<double> v;
    Grid(int n) : N(n), h(1.0/(n+1)), v((n+2)*(n+2), 0.0) {}
    double  operator()(int i, int j) const { return v[i*(N+2) + j]; }
    double& operator()(int i, int j)       { return v[i*(N+2) + j]; }
};

// 矩阵无关 Ax — 与 Jacobi 完全一样
double Ax_at(const Grid& x, int i, int j) {
    double inv_h2 = 1.0 / (x.h * x.h);
    return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
}

// ── RBGS 一次平滑 (原地修改 x) ──
void rbgs_sweep(Grid& x, const Grid& b) {
    double inv_diag = x.h * x.h / 4.0;   // h²/4
    int N = x.N;

    // 阶段 1: 红点 — (i+j) 偶数
    // 红点之间不相邻, 同一阶段内更新顺序无所谓
    for (int i = 1; i <= N; i++)
        for (int j = 1 + (i % 2); j <= N; j += 2)
            x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));

    // 阶段 2: 黑点 — (i+j) 奇数
    // 此时红点已被更新, 黑点的 Ax 会用到新红值 → 收敛更快
    for (int i = 1; i <= N; i++)
        for (int j = 1 + ((i+1) % 2); j <= N; j += 2)
            x(i,j) += inv_diag * (b(i,j) - Ax_at(x, i, j));
}

// ── Jacobi (用于对比) ──
void jacobi_sweep(Grid& x, const Grid& b) {
    double inv_diag = x.h * x.h / 4.0;
    int N = x.N;
    std::vector<double> x_new(x.v.size());

    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            x_new[i*(N+2)+j] = x(i,j) + inv_diag * (b(i,j) - Ax_at(x, i, j));
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            x(i,j) = x_new[i*(N+2)+j];
}

// 计算最大残差
double residual_max(const Grid& x, const Grid& b) {
    double rmax = 0;
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            rmax = std::max(rmax, std::abs(b(i,j) - Ax_at(x, i, j)));
    return rmax;
}

int main(int argc, char* argv[]) {
    int N = 64;
    if (argc > 1) N = std::atoi(argv[1]);

    const double TOL   = 1e-6;
    const int MAX_ITER = 50000;

    std::cout << "=== RBGS vs Jacobi 收敛对比  N=" << N
              << " (" << N*N << " 未知数) ===\n\n";

    // ═══════════════ Jacobi (参照) ═══════════════
    {
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * x.h), sy = std::sin(M_PI * j * x.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }

        auto t0 = std::chrono::high_resolution_clock::now();
        int it = 0;
        for (; it < MAX_ITER; it++) {
            jacobi_sweep(x, b);
            if (it % 200 == 0 && residual_max(x, b) < TOL) break;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Jacobi  " << std::setw(6) << it << " 次  "
                  << s << " s  res=" << residual_max(x, b) << '\n';
    }

    // ═══════════════ RBGS ═══════════════
    {
        Grid x(N), b(N);
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++) {
                double sx = std::sin(M_PI * i * x.h), sy = std::sin(M_PI * j * x.h);
                b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            }

        auto t0 = std::chrono::high_resolution_clock::now();
        int it = 0;
        for (; it < MAX_ITER; it++) {
            rbgs_sweep(x, b);
            if (it % 200 == 0 && residual_max(x, b) < TOL) break;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "RBGS   " << std::setw(6) << it << " 次  "
                  << s << " s  res=" << residual_max(x, b) << '\n';
    }

    std::cout << "\nRBGS 比 Jacobi 快约 2×, 且不需要 x_new 备份数组\n";
    std::cout << "但作为独立求解器仍然太慢 — RBGS 真正的用途是多层网格中的平滑器(Step 4)\n";
    return 0;
}
