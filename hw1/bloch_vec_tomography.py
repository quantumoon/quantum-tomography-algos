import numpy as np
import matplotlib.pyplot as plt


def fidelity(r: np.ndarray, r_exp: np.ndarray) -> np.ndarray:
    r = r.reshape(3,)
    if r_exp.ndim == 1:
        r_exp = r_exp.reshape(3,)

    dot = np.tensordot(r, r_exp, axes=([0], [0]))
    nr2 = float(np.dot(r, r))
    nr_exp2 = np.sum(r_exp * r_exp, axis=0)

    F = 0.5 * (1.0 + dot + np.sqrt(np.maximum(0.0, (1.0 - nr2) * np.maximum(0.0, 1.0 - nr_exp2))))

    return F


def simulate_tomography(r: np.ndarray,
                        n_runs: int = 1000,
                        n_shots: int = 100,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    assert r.shape[0] == 3
    nr = np.linalg.norm(r)
    if nr > 1.0:
        r /= nr

    rng = np.random.default_rng(seed)

    p = (1.0 + r) / 2

    kx = rng.binomial(n_shots, p[0], size=n_runs)
    ky = rng.binomial(n_shots, p[1], size=n_runs)
    kz = rng.binomial(n_shots, p[2], size=n_runs)

    p_exp = np.vstack([kx, ky, kz]) / n_shots
    r_exp = 2.0 * p_exp - 1.0

    F = fidelity(r, r_exp)

    return r, r_exp, F


def plot_results(r: np.ndarray, r_exp: np.ndarray, F: np.ndarray):
    fig = plt.figure(figsize=(9, 5), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.5)
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax2d = fig.add_subplot(gs[0, 1])

    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_surface(X, Y, Z, alpha=0.03, edgecolor='k', linewidth=0.4)

    for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ax3d.plot(np.cos(phi) * np.sin(v), np.sin(phi) * np.sin(v), np.cos(v), lw=0.6, alpha=0.5)
    for lat in np.deg2rad([-60, -30, 0, 30, 60]):
        z0, rxy = np.sin(lat), np.cos(lat)
        ax3d.plot(rxy * np.cos(u), rxy * np.sin(u), z0 * np.ones_like(u), lw=0.6, alpha=0.5)

    L = 1.15
    ax3d.plot([-L, L], [0, 0], [0, 0], lw=1.2, color='black', linestyle='dashed')
    ax3d.text(L, 0, 0, 'x', fontsize=11)
    
    ax3d.plot([0, 0], [-L, L], [0, 0], lw=1.2, color='black', linestyle='dashed')
    ax3d.text(0, L, 0, 'y', fontsize=11)
    
    ax3d.plot([0, 0], [0, 0], [-L, L], lw=1.2, color='black', linestyle='dashed')
    ax3d.text(0, 0, L, '|0⟩', fontsize=11)
    ax3d.text(0, 0, -L * 1.02, '|1⟩', fontsize=11)

    ax3d.quiver(0, 0, 0, *r, arrow_length_ratio=0.2, linewidth=2, color='red', label='True state')
    ax3d.scatter(r_exp[0], r_exp[1], r_exp[2], s=1, alpha=0.5, label='Exp. recovered states')
    ax3d.legend()
    ax3d.set(xlim=[-1.2, 1.2], ylim=[-1.2, 1.2], zlim=[-1.2, 1.2])
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_title('Bloch Sphere')
    ax3d.set_axis_off() 
    ax2d.hist(F, bins='auto')
    ax2d.set_title('Fidelity')
    plt.show()