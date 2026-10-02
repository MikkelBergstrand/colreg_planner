"""Minimal example of the planner API, with a live plot.

    python main.py
"""

import matplotlib.pyplot as plt

from src.planner_api import PredictionService, call_tool

DT = 2.0

# 1. Build the service. Same keys as the `prediction:` block in scenarios/real/*.yaml.
service = PredictionService.from_config(
    {
        "H": 60,                                  # horizon in steps (60 * 2 s = 120 s)
        "interaction": {"use_responding_filter": True, "top_k_hypotheses": 2},
    },
    dt=DT,
    own_id="ego",                                 # own ship: follows COLREGs, not estimated
)


def observe(t: float) -> dict:
    """Ground truth: both vessels hold course (NED metres, degrees, m/s)."""
    return {
        "ego":    {"north": 5.0 * t, "east": 0.0,             "course_deg": 0.0,   "speed": 5.0},
        "target": {"north": 800.0,   "east": 800.0 - 5.0 * t, "course_deg": 270.0, "speed": 5.0},
    }


def draw(ax, history: dict) -> None:
    """Plot the current snapshot using only the API's query functions."""
    ax.clear()
    colors = {"ego": "tab:blue", "target": "tab:red"}

    # Predicted trajectories, one line style per hypothesis (most likely solid).
    for h in service.get_hypotheses(include_trajectories=True, resolution_s=DT):
        for vid, pts in h["trajectories"].items():
            ax.plot([p["east"] for p in pts], [p["north"] for p in pts],
                    color=colors[vid], ls="-" if h["rank"] == 0 else ":",
                    alpha=max(h["probability"], 0.3))

    # Observed track, current position and responsiveness.
    p_resp = service.get_responsiveness()
    for v in service.list_vessels():
        vid = v["vessel_id"]
        history[vid].append((v["east"], v["north"]))
        xs, ys = zip(*history[vid])
        ax.plot(xs, ys, color=colors[vid], lw=3, alpha=0.4)
        ax.plot(v["east"], v["north"], "o", color=colors[vid])
        roles = ", ".join(r["role"] for r in v["roles"]) or "no encounter"
        ax.annotate(f"{vid}\nP(resp)={p_resp[vid]:.2f}\n{roles}",
                    (v["east"], v["north"]), xytext=(8, 8), textcoords="offset points")

    # Predicted closest approach under the most likely hypothesis.
    for c in service.get_predicted_cpa("ego"):
        (n1, e1), (n2, e2) = c["position_vessel"], c["position_other"]
        ax.plot([e1, e2], [n1, n2], "k--", lw=1)
        ax.annotate(f"CPA {c['min_distance_m']:.0f} m\nin {c['t_min_s']:.0f} s",
                    ((e1 + e2) / 2, (n1 + n2) / 2), fontsize=8)

    ax.set_title(f"t = {service.snapshot.t:.0f} s   (solid: most likely hypothesis, dotted: 2nd)")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_xlim(-400, 1000)
    ax.set_ylim(-100, 1300)
    ax.set_aspect("equal")


# 2. Main loop: feed observations once per dt, then query and plot.
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
history = {"ego": [], "target": []}

for k in range(80):
    if not plt.fignum_exists(fig.number):        # window closed -> stop
        break
    t = k * DT
    service.step(observe(t), t=t)
    draw(ax, history)
    plt.pause(0.05)

# 3. Query the final results with plain function calls...
print(service.get_responsiveness())
print(service.get_encounters("ego"))
print(service.get_predicted_cpa("ego"))

# 4. ...or by name, the way an LLM tool call would arrive.
print(call_tool(service, "get_predicted_state", {"vessel_id": "target", "t": 60}))

plt.ioff()
plt.show()
