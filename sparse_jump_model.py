"""
sparse_jump_model.py
─────────────────────────────────────────────────────────────────────────
稀疏跳跃模型（Sparse Jump Model, SJM）— Nystrup et al. (2021) 的一个精简实现。

核心思想（对应研报《基于状态切换信号的动态因子配置》第3.1节）：
  1. Jump Model (JM)：把"状态识别"看成一个"带跳变惩罚的时间序列聚类"问题。
     不像 GMM/K-means 那样逐时点独立判别，JM 在目标函数里显式加入
     "状态切换次数 × jump_penalty" 的惩罚项，用动态规划（DP）联合求解
     整条状态路径，因此状态天然具有持续性、不会像 K-means 那样来回乱跳。
  2. Sparse 扩展：给每个特征一个非负权重 w_j（||w||_2 ≤ 1, ||w||_1 ≤ κ），
     权重由该特征的"类间方差贡献"决定 —— 对区分状态越有效的特征权重越高，
     无效特征权重被压到 0（通过软阈值 + κ 预算实现，做法与 sparse K-means
     [Witten & Tibshirani 2010] 一致）。
  3. 整体是"固定权重拟合 JM → 用 JM 结果重新算特征权重"的交替迭代，直至收敛。

  离线拟合 fit()  用于训练期（可用未来路径做平滑/回看的批量 DP）。
  在线推断 online_predict()  用于样本外：每个新时点只用"截止当天"的信息
     做前向 DP（不做回溯平滑），避免任何未来函数 —— 对应研报中的"在线推理"。
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class OnlineState:
    """单个因子的在线推断状态。"""

    dp: np.ndarray | None = None
    previous_state: int | None = None
    current_date: np.datetime64 | None = None


# ─────────────────────────────────────────────────────────────────────────
# 稀疏投影：给定类间方差贡献 bcss，求满足 ||w||_2<=1, ||w||_1<=kappa, w>=0
# 且使 w·bcss 最大的 w（sparse K-means 的标准子问题，二分搜索软阈值 Δ）
# ─────────────────────────────────────────────────────────────────────────
def _sparse_weight_update(bcss: np.ndarray, kappa: float, tol: float = 1e-6,
                           max_iter: int = 100) -> np.ndarray:
    bcss = np.maximum(bcss, 0.0)
    p = len(bcss)
    if bcss.sum() <= 1e-12:
        return np.ones(p) / np.sqrt(p)

    def soft(delta):
        s = np.maximum(bcss - delta, 0.0)
        norm = np.linalg.norm(s)
        return s / norm if norm > 1e-12 else s

    # Δ=0 时（不做软阈值）如果 L1 范数已经 <= kappa，说明预算充足，直接返回
    w0 = soft(0.0)
    if w0.sum() <= kappa + 1e-9:
        return w0

    lo, hi = 0.0, float(bcss.max())
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        w = soft(mid)
        l1 = w.sum()
        if abs(l1 - kappa) < tol:
            return w
        if l1 > kappa:
            lo = mid
        else:
            hi = mid
    return soft((lo + hi) / 2.0)


# ─────────────────────────────────────────────────────────────────────────
# 带跳变惩罚的状态路径 DP（给定加权距离矩阵 cost[T,K]）
# ─────────────────────────────────────────────────────────────────────────
def _viterbi_path(cost: np.ndarray, jump_penalty: float) -> np.ndarray:
    """批量（可回看）最优路径：min sum cost + penalty*跳变次数"""
    T, K = cost.shape
    dp = np.empty((T, K))
    ptr = np.empty((T, K), dtype=int)
    dp[0] = cost[0]
    for t in range(1, T):
        prev = dp[t - 1]
        best_k = int(np.argmin(prev))
        best_v = prev[best_k]
        for k in range(K):
            stay = prev[k]
            switch = best_v + jump_penalty if best_k != k else \
                     (np.partition(prev, 1)[1] if K > 1 else np.inf) + jump_penalty
            if stay <= switch:
                dp[t, k] = cost[t, k] + stay
                ptr[t, k] = k
            else:
                dp[t, k] = cost[t, k] + switch
                ptr[t, k] = best_k
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmin(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]
    return path


def _online_path(cost: np.ndarray, jump_penalty: float,
                  init_dp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    在线（因果）路径：t 时刻的决策只用 cost[0..t]，不做反向平滑回溯。
    等价于"前向 DP + 逐步取当前最优状态"，可增量调用（传入 init_dp 续算）。
    返回 (每日状态, 最终 dp 向量) 便于滚动调用。
    """
    T, K = cost.shape
    dp = init_dp.copy() if init_dp is not None else cost[0].copy()
    start = 0 if init_dp is None else 0
    path = np.empty(T, dtype=int)
    if init_dp is None:
        path[0] = int(np.argmin(dp))
        start = 1
    for t in range(start, T):
        new_dp = np.empty(K)
        for k in range(K):
            switch_cost = min(dp[j] + jump_penalty for j in range(K) if j != k) if K > 1 else np.inf
            new_dp[k] = cost[t, k] + min(dp[k], switch_cost)
        new_dp -= new_dp.min()  # 数值稳定，不影响 argmin
        dp = new_dp
        path[t] = int(np.argmin(dp))
    return path, dp


class SparseJumpModel:
    """
    参数
    ----
    n_states      : 状态数（市场层建议 3，单因子牛熊建议 2）
    jump_penalty  : 跳变惩罚 λ，越大状态越"粘"、切换越少
    kappa         : 特征稀疏预算 ||w||_1 <= kappa（√p 表示不稀疏／全部特征等权）
    max_outer_iter: 外层（权重 ↔ JM）交替迭代次数
    max_inner_iter: 内层（centroid ↔ path）交替迭代次数
    n_init        : 随机初始化次数，取目标函数最优的一次
    """

    def __init__(self, n_states: int = 2, jump_penalty: float = 30.0,
                 kappa: float | None = None, max_outer_iter: int = 15,
                 max_inner_iter: int = 30, n_init: int = 8, random_state: int = 42):
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.kappa = kappa
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.n_init = n_init
        self.random_state = random_state

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self.state_order_: np.ndarray | None = None  # 按状态均值重排后的映射

    # ── 标准化 ───────────────────────────────────────────────────────────
    def _standardize_fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return (X - self.mean_) / self.std_

    def _standardize(self, X):
        return (X - self.mean_) / self.std_

    # ── 给定权重，拟合一次 Jump Model（内层 centroid↔path 交替）────────────
    def _fit_jm_given_weights(self, X, w, rng):
        T, p = X.shape
        K = self.n_states
        best = None
        for _ in range(self.n_init):
            idx = rng.choice(T, size=K, replace=False)
            centroids = X[idx].copy()
            path = None
            for _ in range(self.max_inner_iter):
                cost = np.zeros((T, K))
                for k in range(K):
                    diff = X - centroids[k]
                    cost[:, k] = (diff ** 2 * w).sum(axis=1)
                new_path = _viterbi_path(cost, self.jump_penalty)
                new_centroids = centroids.copy()
                for k in range(K):
                    mask = new_path == k
                    if mask.any():
                        new_centroids[k] = X[mask].mean(axis=0)
                converged = path is not None and np.array_equal(new_path, path)
                path, centroids = new_path, new_centroids
                if converged:
                    break
            obj = cost[np.arange(T), path].sum() + \
                self.jump_penalty * (path[1:] != path[:-1]).sum()
            if best is None or obj < best[0]:
                best = (obj, path.copy(), centroids.copy())
        return best  # (obj, path, centroids)

    # ── 训练（离线，批量 DP）────────────────────────────────────────────────
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        T, p = X.shape
        Xs = self._standardize_fit(X)
        kappa = self.kappa if self.kappa is not None else np.sqrt(p)
        kappa = max(kappa, 1.0)

        rng = np.random.RandomState(self.random_state)
        w = np.ones(p) / np.sqrt(p)
        path = None
        for outer in range(self.max_outer_iter):
            obj, path, centroids = self._fit_jm_given_weights(Xs, w, rng)

            # 计算每个特征的类间方差贡献 BCSS_j = TSS_j - WCSS_j
            tss = ((Xs - Xs.mean(axis=0)) ** 2).sum(axis=0)
            wcss = np.zeros(p)
            for k in range(self.n_states):
                mask = path == k
                if mask.any():
                    wcss += ((Xs[mask] - centroids[k]) ** 2).sum(axis=0)
            bcss = tss - wcss
            new_w = _sparse_weight_update(bcss, kappa)

            if np.linalg.norm(new_w - w) < 1e-4:
                w = new_w
                break
            w = new_w

        # 按状态均值（第一主特征方向，退化为按整体均值排序更稳健：用状态样本的
        # 加权均值作为排序键，保证跨样本/跨因子编号一致——数值越低=状态编号越小）
        state_score = np.array([
            (centroids[k] * w).sum() for k in range(self.n_states)
        ])
        order = np.argsort(state_score)
        remap = {old: new for new, old in enumerate(order)}

        self.weights_ = w
        self.centroids_ = centroids[order]
        self.state_order_ = order
        self._last_cost_fn_dp_ = None
        return self, np.array([remap[s] for s in path])

    # ── 样本外在线（因果）推断 ─────────────────────────────────────────────
    def online_predict(
        self,
        X_new: np.ndarray,
        online_state: OnlineState | None = None,
        dates: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        对新数据做在线状态推断：t 时刻只使用 <= t 的数据（前向 DP，不回溯）。
        必须在 fit() 之后调用，使用训练期学到的 mean_/std_/weights_/centroids_，
        不重新估计任何参数 —— 保证是真正的 out-of-sample。

        默认保持旧接口语义：每次调用从空 DP 开始。传入 OnlineState 时，
        本次递推从该因子的上一日 DP 继续，并原地更新状态。
        """
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim != 2:
            raise ValueError("X_new必须是二维数组")
        if len(X_new) == 0:
            return np.empty(0, dtype=int)
        if dates is not None and len(dates) != len(X_new):
            raise ValueError("dates长度必须与X_new行数一致")
        date_values = None
        if dates is not None:
            date_values = np.asarray(dates, dtype="datetime64[ns]")
            if np.isnat(date_values).any():
                raise ValueError("dates不能包含无效日期")
            if len(date_values) > 1 and np.any(date_values[1:] <= date_values[:-1]):
                raise ValueError("dates必须严格递增")
            if (
                online_state is not None
                and online_state.current_date is not None
                and date_values[0] <= online_state.current_date
            ):
                raise ValueError("新数据日期必须晚于OnlineState.current_date")

        Xs = self._standardize(X_new)
        K = self.n_states
        cost = np.zeros((len(Xs), K))
        for k in range(K):
            diff = Xs - self.centroids_[k]
            cost[:, k] = (diff ** 2 * self.weights_).sum(axis=1)
        init_dp = online_state.dp if online_state is not None else None
        path, final_dp = _online_path(cost, self.jump_penalty, init_dp=init_dp)
        if online_state is not None:
            online_state.dp = final_dp.copy()
            online_state.previous_state = int(path[-1])
            if date_values is not None:
                online_state.current_date = date_values[-1]
        return path  # 已经是按 fit() 时排好序的状态编号（0=均值最低...）


# ─────────────────────────────────────────────────────────────────────────
# 自检：合成一段有明显状态切换的数据，检验 SJM 能否恢复真实状态
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    T = 600
    true_state = np.zeros(T, dtype=int)
    true_state[150:300] = 1
    true_state[300:420] = 2
    true_state[420:] = 1
    means = {0: [-0.02, 0.03, -1.0], 1: [0.0, 0.01, 0.0], 2: [0.03, -0.01, 1.5]}
    X_informative = np.array([means[s] for s in true_state]) + rng.normal(0, 0.5, (T, 3))
    X_noise = rng.normal(0, 1.0, (T, 5))  # 5 个无信息噪声特征
    X = np.hstack([X_informative, X_noise])

    sjm = SparseJumpModel(n_states=3, jump_penalty=8.0, kappa=3.0, n_init=6)
    sjm, path_train = sjm.fit(X[:400])
    print("训练集特征权重:", np.round(sjm.weights_, 3))
    print("  (前3维为有效特征，后5维应接近0)")

    online_path = sjm.online_predict(X[400:])
    acc_online = (online_path == true_state[400:]).mean()
    print(f"在线样本外状态识别与真实状态吻合率: {acc_online:.1%}")
    acc_train = (path_train == true_state[:400]).mean()
    print(f"训练集(批量)状态识别与真实状态吻合率: {acc_train:.1%}")
