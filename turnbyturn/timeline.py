import time

import matplotlib.pyplot as plt
import numpy as np

LOG = {"pc": [], "sota": [], "speaker": [], "gate_lag": []}  # (time, value)


def rms(raw):
    x = np.frombuffer(raw, dtype=np.int16) / 32768
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def mark(source, value):
    LOG[source].append((time.time(), value))


def _first_above(series, start, level=0.03):
    for t, v in series:
        if t >= start and v > level:
            return t
    return None


def plot_timeline(path="mic_delay.png"):
    pc, sota, spk = (list(LOG[k]) for k in ("pc", "sota", "speaker"))
    if not (pc and sota and spk):
        print("Not enough data.")
        return
    t0 = min(pc[0][0], sota[0][0])

    # speech segments sent to sota
    segments, start = [], None
    for t, on in spk:
        if on and start is None:
            start = t
        elif not on and start is not None:
            segments.append((start, t))
            start = None

    # per turn: when each mic first hears sota after sending starts
    print("\nturn  sent at   PC mic hears   Sota mic hears   Sota mic delay")
    for i, (s, e) in enumerate(segments, 1):
        p = _first_above(pc, s)
        q = _first_above(sota, s)
        if p and q:
            print(
                f"{i:>4}  {s - t0:7.2f}  {p - s:+10.2f} s  {q - s:+12.2f} s  {q - p:12.2f} s"
            )

    lag = list(LOG["gate_lag"])
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for ax, data, label, color in (
        (axes[0], pc, "PC mic (real time)", "#2a78d6"),
        (axes[1], sota, "Sota mic (via robot)", "#eb6834"),
    ):
        for s, e in segments:
            ax.axvspan(s - t0, e - t0, color="#9a9a9a", alpha=0.25, linewidth=0)
        ax.plot(
            [t - t0 for t, _ in data], [v for _, v in data], color=color, linewidth=1
        )
        ax.set_ylabel("RMS")
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(alpha=0.2)

    for s, e in segments:
        axes[2].axvspan(s - t0, e - t0, color="#9a9a9a", alpha=0.25, linewidth=0)
    axes[2].plot(
        [t - t0 for t, _ in lag], [v for _, v in lag], color="#1baf7a", linewidth=1
    )
    axes[2].set_ylabel("seconds")
    axes[2].set_title(
        "Gate lag: age of each chunk when the gate sees it", loc="left", fontsize=10
    )
    axes[2].grid(alpha=0.2)
    axes[2].set_xlabel("Time (s)")

    axes[0].plot(
        [], [], color="#9a9a9a", alpha=0.5, linewidth=8, label="Speech sent to Sota"
    )
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.show()
