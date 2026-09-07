"""Reference: turn a policy action chunk into a smooth command stream at the controller's rate.

Hand this to whoever writes the G2 control loop. It is numpy-only and has no openpi dependency, so
it can be lifted into a control stack as-is.

WHAT GOES WRONG WITHOUT IT
--------------------------
The policy emits 50 points per chunk, and the obvious thing -- send one point per control tick -- is
wrong twice over, which is what the stutter is.

First, the points are not evenly spaced in time. They are placed by CURVATURE: density follows
|d2q|^(1/3), so the trajectory gets points where it bends and skips ahead where it runs straight.
Measured over eight episodes, consecutive points differ by

    joint distance   p05 0.0108   p50 0.0685   p95 0.1641 rad     -- a factor of 15
    elapsed time     p05 1.44     p50 2.92     p95 4.95   frames  -- a factor of 3.4

so issuing one point per tick swings the joint speed across a 15x range. No amount of smoothing
downstream fixes that; the timing itself has to come from the geometry.

Second, even with correct per-point timing, jumping from point to point leaves the velocity
discontinuous at every point. That is what the interpolation below is for.

HOW TIME IS ASSIGNED
--------------------
    dt_i = max(|q_{i+1} - q_i| / v_nominal, dt_min)

Distance over a nominal speed, with a floor. The floor is not a safety fudge: inside a suction
transient the points are deliberately close together in space but stand for real elapsed time -- the
vacuum needs those milliseconds whether or not the arm moves -- and a pure distance rule would blast
through the seal. Checked against ten recordings, this rule reproduces the demonstrated duration:

    v_nominal 0.60, dt_min 0.04  ->  1.075x the recording
    v_nominal 0.75, dt_min 0.06  ->  0.903x          <- the defaults here
    v_nominal 0.90, dt_min 0.06  ->  0.782x

v_nominal is therefore the single knob for overall pace; raising it speeds the whole cycle up
proportionally, and the limiter below keeps the result inside the machine's envelope.

CHUNK SEAMS AND LATENCY
-----------------------
`latency_s` is continuous, not a whole number of frames. A chunk computed from an observation 80 ms
ago is entered at tau = 0.080 s along its own timeline, between points, which is exactly what the
interpolation makes possible -- there is no reason to round to the nearest point and inherit the
error.

Successive chunks are anchored on different observations and need not agree. Measured on our data
the disagreement is small (p50 0.055 rad, about one normal point spacing) but it is not zero, and
switching hard would step the velocity. So the previous chunk stays live for `blend_s` and the two
are cross-faded with a smoothstep, whose derivative vanishes at both ends -- the blend is therefore
velocity-continuous even though the two chunks are not.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python chunk_playback.py --demo
"""

from __future__ import annotations

import dataclasses

import numpy as np

# ---------------------------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------------------------


def _fritsch_carlson(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Tangents for a cubic Hermite that cannot overshoot between its own knots.

    Plain cubic splines ring: to stay smooth they overshoot on either side of a sharp move, and an
    overshoot here is the arm going somewhere the policy never asked for. Limiting the tangents the
    Fritsch-Carlson way keeps every interval monotone, so the command stays inside the envelope the
    points define. This is the same shape-preserving rule the training labels were built with.
    """
    steps = np.diff(times)[:, None]
    slopes = np.diff(values, axis=0) / steps
    tangents = np.zeros_like(values)
    tangents[1:-1] = 0.5 * (slopes[:-1] + slopes[1:])
    tangents[0], tangents[-1] = slopes[0], slopes[-1]

    # Where consecutive slopes disagree in sign the point is a local extremum: flatten it.
    sign_change = slopes[:-1] * slopes[1:] <= 0
    tangents[1:-1][sign_change] = 0.0
    # Elsewhere clamp to three times the smaller neighbouring slope, the monotonicity condition.
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in ((0, -1), (1, None)):
            neighbour = slopes[lo:hi] if hi is not None else slopes[lo:]
            bound = 3.0 * np.abs(neighbour)
            excess = np.abs(tangents[1:-1]) > bound
            tangents[1:-1] = np.where(excess, np.sign(tangents[1:-1]) * bound, tangents[1:-1])
    return tangents


@dataclasses.dataclass
class Hermite:
    """A C1 cubic through the chunk points, evaluable at any time, with its derivative."""

    times: np.ndarray  # (n,) seconds along this chunk's own timeline, strictly increasing
    values: np.ndarray  # (n, dof)
    tangents: np.ndarray  # (n, dof)

    @classmethod
    def through(cls, times: np.ndarray, values: np.ndarray) -> Hermite:
        return cls(times, values, _fritsch_carlson(times, values))

    def at(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity at `tau`, clamped to the ends rather than extrapolated.

        Extrapolating a cubic past its last knot diverges fast, and past the end of a chunk is
        exactly where a late replan leaves the controller. Holding the final pose is the safe
        reading of "the plan ran out".
        """
        if tau <= self.times[0]:
            return self.values[0].copy(), np.zeros_like(self.values[0])
        if tau >= self.times[-1]:
            return self.values[-1].copy(), np.zeros_like(self.values[-1])

        i = int(np.searchsorted(self.times, tau) - 1)
        h = self.times[i + 1] - self.times[i]
        t = (tau - self.times[i]) / h
        p0, p1 = self.values[i], self.values[i + 1]
        m0, m1 = self.tangents[i] * h, self.tangents[i + 1] * h

        t2, t3 = t * t, t * t * t
        position = (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1
        velocity = ((6 * t2 - 6 * t) * p0 + (3 * t2 - 4 * t + 1) * m0
                    + (-6 * t2 + 6 * t) * p1 + (3 * t2 - 2 * t) * m1) / h
        return position, velocity


# ---------------------------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------------------------


def schedule(points: np.ndarray, v_nominal: float, dt_min: float) -> np.ndarray:
    """Seconds at which each chunk point should be reached, measured from the observation instant.

    Point 0 already sits one step ahead of the observation -- the labels are built with k = 1..H --
    so the first entry is dt_0 rather than zero.
    """
    steps = np.maximum(np.linalg.norm(np.diff(points, axis=0), axis=1) / v_nominal, dt_min)
    first = max(float(np.linalg.norm(points[0] - points[min(1, len(points) - 1)])) / v_nominal, dt_min)
    return np.cumsum(np.concatenate([[first], steps]))


# ---------------------------------------------------------------------------------------------
# Rate limiting -- the motion-control half
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Limits:
    v_max: np.ndarray
    a_max: np.ndarray
    j_max: np.ndarray


class RateLimiter:
    """Follow a reference stream without exceeding velocity, acceleration or jerk, per joint.

    The interpolation above already removes the discontinuities the policy's own output would cause.
    This is the guard for everything else: a chunk that disagrees with its predecessor, a replan that
    arrives late, a policy that emits a step the arm cannot physically take. It clamps jerk first,
    then acceleration, then velocity, so the acceleration the command implies is always reachable
    from the one before it.
    """

    def __init__(self, position: np.ndarray, limits: Limits, dt: float, settle_s: float = 0.10) -> None:
        self.position = position.astype(np.float64).copy()
        self.velocity = np.zeros_like(self.position)
        self.acceleration = np.zeros_like(self.position)
        self.limits, self.dt = limits, dt
        # Gain on the tracking error, as a settling time rather than a bare number: the command
        # closes a standing offset in roughly this long. Expressing it as 1/dt instead -- close the
        # whole gap in one tick -- turns the limiter into a chaser that saturates at v_max and
        # arrives early, which is what the first version did.
        self.gain = 1.0 / settle_s

    def step(self, target: np.ndarray, target_velocity: np.ndarray) -> np.ndarray:
        dt = self.dt
        # Feed-forward the reference's own velocity, correct only the offset. The limiter then rides
        # the trajectory rather than pulling toward its position.
        wanted = target_velocity + (target - self.position) * self.gain
        wanted = np.clip(wanted, -self.limits.v_max, self.limits.v_max)

        wanted_acceleration = (wanted - self.velocity) / dt
        jerk = np.clip((wanted_acceleration - self.acceleration) / dt, -self.limits.j_max, self.limits.j_max)
        self.acceleration = np.clip(self.acceleration + jerk * dt, -self.limits.a_max, self.limits.a_max)
        self.velocity = np.clip(self.velocity + self.acceleration * dt, -self.limits.v_max, self.limits.v_max)
        self.position = self.position + self.velocity * dt
        return self.position.copy()


# ---------------------------------------------------------------------------------------------
# The online follower
# ---------------------------------------------------------------------------------------------


def _smoothstep(x: float) -> float:
    """3x^2 - 2x^3: zero derivative at both ends, which is what makes the cross-fade C1."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclasses.dataclass
class Playback:
    exec_hz: float = 100.0
    v_nominal: float = 0.75  # rad/s in joint space; the single pace knob
    dt_min: float = 0.06  # seconds per point floor, so suction transients keep their duration
    blend_s: float = 0.15  # cross-fade window when a new chunk arrives
    limits: Limits | None = None


class ChunkFollower:
    """Holds the active chunk, advances its clock, and hands over to the next one smoothly."""

    def __init__(self, config: Playback) -> None:
        self.config = config
        self.dt = 1.0 / config.exec_hz
        self.active: Hermite | None = None
        self.previous: Hermite | None = None
        self.tau = 0.0  # seconds along the active chunk's timeline
        self.tau_previous = 0.0
        self.blend_left = 0.0
        self.limiter: RateLimiter | None = None

    def push(self, chunk: np.ndarray, latency_s: float = 0.0) -> None:
        """Accept a chunk computed from an observation `latency_s` in the past.

        `latency_s` is continuous. Rounding it to whole points would throw away up to half a point of
        position for no reason -- the interpolation is precisely what makes a fractional entry
        point available.
        """
        times = schedule(chunk, self.config.v_nominal, self.config.dt_min)
        fresh = Hermite.through(times, chunk.astype(np.float64))

        if self.active is not None:
            self.previous, self.tau_previous = self.active, self.tau
            self.blend_left = self.config.blend_s
        self.active, self.tau = fresh, float(latency_s)

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """One control tick: the commanded position and the velocity it implies."""
        if self.active is None:
            raise RuntimeError("push() a chunk before stepping")

        position, velocity = self.active.at(self.tau)
        if self.previous is not None and self.blend_left > 0.0:
            old_position, old_velocity = self.previous.at(self.tau_previous)
            weight = _smoothstep(1.0 - self.blend_left / self.config.blend_s)
            position = (1.0 - weight) * old_position + weight * position
            velocity = (1.0 - weight) * old_velocity + weight * velocity
            self.tau_previous += self.dt
            self.blend_left -= self.dt
            if self.blend_left <= 0.0:
                self.previous = None

        if self.config.limits is not None:
            if self.limiter is None:
                self.limiter = RateLimiter(position, self.config.limits, self.dt)
            position = self.limiter.step(position, velocity)
            velocity = self.limiter.velocity.copy()

        self.tau += self.dt
        return position, velocity

    @property
    def exhausted(self) -> bool:
        return self.active is not None and self.tau > self.active.times[-1]
