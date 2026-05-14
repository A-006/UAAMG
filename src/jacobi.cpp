// ===========================================================================
// Step 1: Jacobi 迭代求解 2D Poisson 方程
// 编译: g++ -std=c++17 -O2 jacobi.cpp -o jacobi
// 运行: ./jacobi [N]          (默认 N=64)
// ===========================================================================
//
// 问题: 在单位正方形 [0,1]×[0,1] 上求解
//      -Δu = f       (Poisson 方程)
//      u = 0 在边界   (Dirichlet 边界条件)
//
// 验证用的已知解: u(x,y) = sin(πx)·sin(πy)
// 代入得右端项:   f(x,y) = 2π²·sin(πx)·sin(πy)
//
// ===========================================================================
// 【从连续到离散 — 5 点 Laplacian 模板的推导】
// ===========================================================================
//
// 二阶导数的中心差分近似 (h 为网格间距):
//   u''(x) ≈ [u(x+h) - 2u(x) + u(x-h)] / h²
//   这个公式的误差是 O(h²), 即二阶精度
//
// 二维 Laplace 算子 Δu = ∂²u/∂x² + ∂²u/∂y², 在均匀网格上:
//
//   Δu|_{i,j} ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / h²     ← x方向
//             + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / h²     ← y方向
//
//   合并:  (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
//
//   加负号 (Poisson 是 -Δu = f):
//
//   (-Δu)|_{i,j} ≈ (4u_{i,j} - u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1}) / h²
//
//   这就是代码中 Ax_at() 的公式 — 5 点模板, 只用自己和 4 个邻居
//
//   矩阵 A 的结构:
//     对角线 =  4/h²    ← 自己的系数
//     邻居   = -1/h²    ← 上/下/左/右的系数
//     其余   =  0       ← 不相邻的点没有直接联系
//
// ===========================================================================
// 【Jacobi 迭代公式】
// ===========================================================================
//
//   Ax = b  →  把 A 拆成 D + (A-D), D=对角线
//   Dx = b - (A-D)x
//   x^{new} = D⁻¹(b - A x^{old})         ← Jacobi 迭代
//
//   D⁻¹ = h²/4  (对角线的倒数: 1 ÷ (4/h²))
//
//   每个格点: x_{i,j}^{new} = x_{i,j}^{old} + (h²/4)·[b_{i,j} - (Ax)^{old}_{i,j}]
//                                                   └────── 残差 ──────────┘
//
//   直观: 残差>0 → 当前值偏小, 加一点; 残差<0 → 当前值偏大, 减一点
//
//   特点: 所有点用旧值同步更新 → 天然可并行
//   缺点: 信息每轮只传 1 格 → 大网格收敛极慢
// ===========================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// ── 网格: 二维数组 (N+2)×(N+2), 展平为一维存储 ──
// 索引: (i,j) → i*(N+2)+j
// 边界行/列 (i=0, N+1, j=0, N+1) 存边界条件, 不参与迭代
struct Grid {
    int N;                     // 内部格点数
    double h;                  // 网格间距 = 1/(N+1)
    std::vector<double> v;     // 数据, 长度 (N+2)*(N+2)

    Grid(int n) : N(n), h(1.0/(n+1)), v((n+2)*(n+2), 0.0) {}

    double  operator()(int i, int j) const { return v[i*(N+2) + j]; }
    double& operator()(int i, int j)       { return v[i*(N+2) + j]; }
};

// ── 矩阵无关的 A*x — 5 点 Laplacian ──
// 不存储 A, 直接从邻居值计算 (Ax)_{i,j}
double Ax_at(const Grid& x, int i, int j) {
    double inv_h2 = 1.0 / (x.h * x.h);  // 1/h²
    return (4*x(i,j) - x(i+1,j) - x(i-1,j) - x(i,j+1) - x(i,j-1)) * inv_h2;
}

int main(int argc, char* argv[]) {
    int N = 64;
    if (argc > 1) N = std::atoi(argv[1]);

    const double TOL   = 1e-6;
    const int MAX_ITER = 50000;
    double inv_diag    = 1.0 / ((N+1.0)*(N+1.0)) / 4.0;  // h²/4 = D⁻¹

    Grid x(N);   // 解, 初始全 0
    Grid b(N);   // 右端项 f
    Grid u(N);   // 真解, 最后验证用

    // ── 初始化右端项和真解 ──
    // 真解 u=sin(πx)sin(πy) → f=-Δu=2π²·sin(πx)·sin(πy)
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++) {
            double sx = std::sin(M_PI * i * x.h);
            double sy = std::sin(M_PI * j * x.h);
            b(i,j) = 2.0 * M_PI * M_PI * sx * sy;
            u(i,j) = sx * sy;
        }

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> x_new(x.v.size());

    // ═══════════════ Jacobi 主循环 ═══════════════
    for (int iter = 0; iter < MAX_ITER; iter++) {

        // 所有点用 x_old 同步算 x_new (Jacobi 特征)
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                x_new[i*(N+2)+j] = x(i,j) + inv_diag * (b(i,j) - Ax_at(x, i, j));

        // 写回 x
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                x(i,j) = x_new[i*(N+2)+j];

        // 每 200 轮报告并检查收敛
        if (iter % 200 == 0) {
            double rmax = 0;
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++)
                    rmax = std::max(rmax, std::abs(b(i,j) - Ax_at(x, i, j)));

            std::cout << "iter " << std::setw(6) << iter
                      << "  res = " << rmax << '\n';

            if (rmax < TOL) {
                auto t1 = std::chrono::high_resolution_clock::now();
                double s = std::chrono::duration<double>(t1 - t0).count();

                double err = 0;
                for (int i = 1; i <= N; i++)
                    for (int j = 1; j <= N; j++)
                        err = std::max(err, std::abs(x(i,j) - u(i,j)));

                std::cout << "\n收敛: " << iter << " 次, "
                          << s << " s, 误差 " << err << '\n';
                return 0;
            }
        }
    }
    std::cout << "未收敛 (N=" << N << ")\n";
    return 1;
}
