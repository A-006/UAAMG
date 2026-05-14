// ===========================================================================
// Step 3: CG (Conjugate Gradient) — 共轭梯度法
// 编译: g++ -std=c++17 -O2 cg.cpp -o cg
// 运行: ./cg [N]
// ===========================================================================
//
// 【为什么 Jacobi/RBGS 不够 —— 局部方法的瓶颈】
// Jacobi 和 RBGS 每轮只更新一个点, 只看自己和邻居
//   → 高频(局部)误差消除快
//   → 低频(全局)误差几乎不动 —— 比如整个区域偏大的"大波浪"误差
//   网格越大, 低频误差越多 → 收敛随 N 增大而急剧恶化
//
// 【CG 的思路: 每次选择全局最优的搜索方向】
// 不逐点更新, 而是在 N² 维空间中沿"搜索方向"前进。
// 关键是: 每次选的新方向 p_k 与之前所有方向 p_0...p_{k-1} 都 A-共轭
//
//   A-共轭: p_i^T A p_j = 0  (i≠j)
//
// 这意味着什么? 每个方向只走一次, 不会在同一方向上重复做功。
// n 个 A-共轭方向张成整个空间 → 理论 n 步内精确收敛 (n = 未知数个数)
// 实际上几十到上百步就够用。
//
// ===========================================================================
// 【CG 算法 — 每轮只需 1 次矩阵乘向量 + 2 次内积】
// ===========================================================================
//
// x₀ = 0                         初始猜测(任意)
// r₀ = b - Ax₀                   残差 (误差的度量)
// p₀ = r₀                        第一个搜索方向 = 残差方向(最速下降)
//
// for k = 0,1,2,...:
//   αₖ = (rₖ·rₖ) / (pₖ·Apₖ)     沿 pₖ 方向的最优步长
//   x_{k+1} = xₖ + αₖ·pₖ         更新解
//   r_{k+1} = rₖ - αₖ·Apₖ        更新残差 (注: Apₖ 可复用!)
//   βₖ = (r_{k+1}·r_{k+1})/(rₖ·rₖ)
//   p_{k+1} = r_{k+1} + βₖ·pₖ    下一个搜索方向 (A-共轭修正)
//
// 两个关键计算:
//   matvec(p): 矩阵乘向量, 用 5 点模板 → 矩阵无关
//   dot(a,b):  向量内积, 只算内部格点
//
// ===========================================================================
// 【CG vs Jacobi/RBGS —— 本质区别】
// ===========================================================================
// Jacobi/RBGS: x_{i,j}^{new} = x_{i,j} + ω*(b_{i,j} - Ax_{i,j})
//              每点独立修正, 没有全局协调
//
// CG:         每次选一个方向 p, 沿 p 走到该方向的精确极小值
//              方向之间互相 A-共轭 → 每个方向只走一次, 不走回头路
//
// 代价: CG 每轮要做内积(全局通信), 而 Jacobi 每轮不需要
//       但 CG 轮数远少于 Jacobi → 总时间更短
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

// ── A*v — 5 点 Laplacian ──
// 返回一个新 Grid, 不修改输入
Grid matvec(const Grid& v) {
    int N = v.N;
    double inv_h2 = 1.0 / (v.h * v.h);
    Grid Av(N);
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            Av(i,j) = (4*v(i,j) - v(i+1,j) - v(i-1,j) - v(i,j+1) - v(i,j-1)) * inv_h2;
    return Av;
}

// ── 内积 — 仅内部格点 ──
double dot(const Grid& a, const Grid& b) {
    double s = 0;
    for (int i = 1; i <= a.N; i++)
        for (int j = 1; j <= a.N; j++)
            s += a(i,j) * b(i,j);
    return s;
}

// ── y += a * x ──
void axpy(double a, const Grid& x, Grid& y) {
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            y(i,j) += a * x(i,j);
}

// ── y = a * x + y  (和 axpy 一样, 这是 CG 里另一种用法) ──
// 在 CG 中 p = r + beta * p 用这个: 先缩放旧 p, 再加 r
void xpay(const Grid& x, double a, Grid& y) {
    for (int i = 1; i <= x.N; i++)
        for (int j = 1; j <= x.N; j++)
            y(i,j) = x(i,j) + a * y(i,j);
}

int main(int argc, char* argv[]) {
    int N = 64;
    if (argc > 1) N = std::atoi(argv[1]);

    const double TOL   = 1e-8;
    const int MAX_ITER = 5000;

    std::cout << "=== CG 共轭梯度法  N=" << N
              << " (" << N*N << " 未知数)  tol=" << TOL << " ===\n\n";

    Grid x(N);   // 解
    Grid b(N);   // 右端项

    // 已知解 u = x(1-x)·y(1-y)  (非特征函数, CG需要多轮)
    // f = -Δu = 2x(1-x) + 2y(1-y)
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++) {
            double xi = i * x.h, yj = j * x.h;
            b(i,j) = 2.0 * (xi*(1-xi) + yj*(1-yj));
        }

    // ═══════════════ CG 主循环 ═══════════════
    auto t0 = std::chrono::high_resolution_clock::now();

    // 初始化: r = b - A*0 = b,  p = r
    Grid r = b;        // 残差 (复制 b)
    Grid p = r;        // 搜索方向
    double rsold = dot(r, r);         // (r₀, r₀)
    double r0_norm = std::sqrt(rsold); // ||r₀||, 用于显示相对残差

    for (int k = 0; k < MAX_ITER; k++) {
        // ── α = (r·r) / (p·Ap) ──
        Grid Ap = matvec(p);              // 矩阵乘向量 (最耗时的操作)
        double pAp = dot(p, Ap);          // (p, Ap)
        double alpha = rsold / pAp;       // 沿 p 方向的最优步长

        // ── x = x + α*p,  r = r - α*Ap ──
        axpy(alpha, p, x);               // x += α * p
        axpy(-alpha, Ap, r);             // r -= α * Ap  (复用刚才的 Ap!)

        // ── 检查收敛 ──
        double rsnew = dot(r, r);
        if (k % 10 == 0)
            std::cout << "iter " << std::setw(4) << k
                      << "  |r|/|r0| = " << std::sqrt(rsnew) / r0_norm << '\n';

        if (std::sqrt(rsnew) < TOL) {  // 绝对残差 < tol
            auto t1 = std::chrono::high_resolution_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();

            double err = 0;
            for (int i = 1; i <= N; i++)
                for (int j = 1; j <= N; j++) {
                    double xi = i * x.h, yj = j * x.h;
                    err = std::max(err, std::abs(x(i,j) - xi*(1-xi)*yj*(1-yj)));
                }

            std::cout << "\n收敛: " << k << " 次迭代, " << s << " s, 误差 " << err << '\n';
            return 0;
        }

        // ── β = (r_new·r_new) / (r_old·r_old) ──
        double beta = rsnew / rsold;

        // ── p = r + β*p (新方向 = 残差 + 旧方向的共轭修正) ──
        // 先缩放旧 p, 再加 r_new
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                p(i,j) = r(i,j) + beta * p(i,j);

        rsold = rsnew;
    }

    std::cout << "未收敛\n";
    return 1;
}
