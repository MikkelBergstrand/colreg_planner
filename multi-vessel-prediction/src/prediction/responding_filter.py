import math
import numpy as np
from scipy.special import ive


_EPS: float = 1e-9


class RespondingFilter:
    """Bayesian responsiveness filter (paper Section IV-C).

    Tracks r_t in {R, N} per vessel via the recursion in Eq. 5:

        p(r_t | z_1:t) ∝ p(z_t | r_t) * Σ_{r_t-1} p(r_t | r_t-1) p(r_t-1 | z_1:t-1)

    Transition model (Eqs. 6-7): a 2-state Markov chain with
    epsilon_R ≜ p(r_t=N | r_t-1=R) (``epsilon_c`` here) and epsilon_N ≜
    p(r_t=R | r_t-1=N) derived so the stationary distribution equals the
    prior p(r0=R).

    Observation likelihood (Eqs. 8-9): course-only von Mises density on the
    observed vs. predicted heading, chosen over the raw (vn, ve) residual
    because course changes are favoured over speed reduction under Rule 8
    (see _course_likelihood). L_R marginalizes over the responsive-branch
    hypotheses (the VO-rollout predictions passed to prepare_prediction),
    weighted by each hypothesis's predicted probability; L_N is a single
    von Mises centred on the nominal-trajectory prediction.

    Usage: at each timestep call step(z_obs) to update the belief using the
    predictions staged by last step's prepare_prediction(), then call
    prepare_prediction(...) again to stage next step's predictions.
    """

    def __init__(
        self,
        p_responsive_init: float = 0.99,
        epsilon_c        : float = 1e-5,
        sigma_theta_deg  : float = 3.0,
        min_speed_for_course: float = 0.5,
        warmup_steps     : int   = 10,
    ) -> None:
        """
        Parameters
        ----------
        p_responsive_init : float
            Prior p(r0 = R). Also fixes the transition matrix's stationary
            distribution, so the belief reverts here when evidence is
            neutral (paper's p(r0=R) = 0.99).
        epsilon_c : float
            epsilon_R in the paper: p(r_t=N | r_t-1=R) per step. epsilon_N
            (N -> R) is derived via Eq. 7 so the stationary distribution
            equals p_responsive_init.
        sigma_theta_deg : float
            Course noise std [deg] (paper's sigma_psi); the von Mises
            concentration is kappa = 1 / sigma_psi_rad**2 (Eq. 8).
        min_speed_for_course : float
            Speed [m/s] below which course is undefined/unreliable (heading
            of a near-zero velocity vector is noise, not signal) — the
            course likelihood is neutralized (set to 1) below this. Not in
            the paper; a numerical guard for near-stationary vessels.
        warmup_steps : int
            Initial steps during which the belief is held at its prior, so
            the COLREGs classifier's hysteresis has time to stabilize
            before the first update.
        """
        self.responding_state  = np.array([p_responsive_init, 1.0 - p_responsive_init])
        self._p_init           = p_responsive_init
        self.epsilon_c         = epsilon_c
        # epsilon_n derived so stationary dist = prior: ε_n/(ε_c+ε_n) = p_init
        # Clamped to [0, 1] so it remains a valid probability.
        self.epsilon_n         = min(epsilon_c * p_responsive_init / max(1.0 - p_responsive_init, _EPS), 1.0)
        self.transition_matrix = self._build_transition_matrix()

        self._warmup_steps     = warmup_steps
        self._step_count       = 0

        self._kappa                 = 1.0 / math.radians(sigma_theta_deg) ** 2
        self._min_speed_for_course  = min_speed_for_course

        # Staged prediction storage: _next is written by prepare_prediction(),
        # then swapped into the active slot at the start of the next step().
        # _z_pred_vo is a list of (weight, z_pred) tuples; weights sum to 1.
        self._z_pred_vo:       list[tuple[float, np.ndarray]] | None = None
        self._z_pred_nom:      np.ndarray | None                     = None
        self._z_pred_vo_next:  list[tuple[float, np.ndarray]] | None = None
        self._z_pred_nom_next: np.ndarray | None                     = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_transition_matrix(self) -> np.ndarray:
        return np.array([
            [1 - self.epsilon_c, self.epsilon_n],
            [self.epsilon_c,     1 - self.epsilon_n],
        ])

    def _course_likelihood(self, z_obs: np.ndarray, z_pred: np.ndarray) -> float:
        """Von Mises density on the observed vs. predicted course (Eq. 8).

        Scores heading only, not the full kinematic state: course changes
        give an immediate signal of response and are favoured over speed
        reduction by Rule 8, so a vessel responding solely through speed
        would otherwise be missed.
        """
        v_obs, v_pred = z_obs[2:4], z_pred[2:4]
        speed_obs, speed_pred = float(np.linalg.norm(v_obs)), float(np.linalg.norm(v_pred))

        if min(speed_obs, speed_pred) < self._min_speed_for_course:
            # Heading of a near-zero velocity vector is noise, not signal.
            return 1.0

        delta = math.atan2(v_obs[1], v_obs[0]) - math.atan2(v_pred[1], v_pred[0])
        kappa = self._kappa
        # exp(kappa*(cos(delta)-1)) / (2*pi*ive(0, kappa)) is the
        # exponentially-rescaled von Mises pdf: numerically stable for
        # large kappa (ive = i0(kappa) * exp(-kappa), never overflows).
        return math.exp(kappa * (math.cos(delta) - 1.0)) / (2.0 * math.pi * ive(0, kappa))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_prediction(
        self,
        responsive_preds: list[tuple[float, np.ndarray]],
        z_pred_nom:       np.ndarray,
    ) -> None:
        """Store pre-computed 1-step-ahead predictions for the next step().

        Call this AFTER step() each timestep so the stored predictions are
        ready for comparison at the following observation.

        Parameters
        ----------
        responsive_preds : list of (weight, ndarray(4,))
            Per-hypothesis (weight, [north, east, vn, ve]) predictions for the
            responsive branch. Weights are normalized to sum to 1 by the caller.
        z_pred_nom : ndarray(4,)
            Nominal [north, east, vn, ve] prediction for the non-responsive branch.
        """
        if not responsive_preds:
            return

        total = sum(w for w, _ in responsive_preds)
        if total > _EPS:
            self._z_pred_vo_next = [(w / total, z) for w, z in responsive_preds]
        else:
            n = len(responsive_preds)
            self._z_pred_vo_next = [(1.0 / n, z) for _, z in responsive_preds]
        self._z_pred_nom_next = z_pred_nom

    def step(self, z_obs: np.ndarray) -> np.ndarray:
        """Full filter step: swap staged predictions, then predict + update.

        Parameters
        ----------
        z_obs : ndarray(4,)
            Current observed joint state [north, east, vn, ve].

        Returns
        -------
        ndarray(2,)
            Updated belief [p_responsive, p_non_responsive].
        """
        self._z_pred_vo  = self._z_pred_vo_next
        self._z_pred_nom = self._z_pred_nom_next
        self._step_count += 1

        if self._z_pred_vo is None or self._step_count <= self._warmup_steps:
            return self.responding_state

        predicted = self.transition_matrix @ self.responding_state

        # L_R: marginalize over the responsive-branch hypotheses (Eq. 9).
        L_R = sum(
            w * self._course_likelihood(z_obs, z_pred)
            for w, z_pred in self._z_pred_vo
        )
        L_N = self._course_likelihood(z_obs, self._z_pred_nom)

        likelihoods  = np.array([max(L_R, _EPS), max(L_N, _EPS)])
        unnormalized = likelihoods * predicted
        total        = float(unnormalized.sum())
        if total < _EPS:
            self.responding_state = predicted
        else:
            self.responding_state = unnormalized / total

        return self.responding_state

    def reset(self, p_responsive_init: float | None = None) -> None:
        """Reset belief and clear stored predictions."""
        p = p_responsive_init if p_responsive_init is not None else self._p_init
        self.responding_state  = np.array([p, 1.0 - p])
        self._z_pred_vo        = None
        self._z_pred_nom       = None
        self._z_pred_vo_next   = None
        self._z_pred_nom_next  = None
        self._step_count       = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def p_responsive(self) -> float:
        return float(self.responding_state[0])

    def predicted_p_responsive(self) -> float:
        """One-step-ahead predicted P(responsive): p̄(r_t) from Eq. 9,
        obtained by propagating the current posterior through the
        transition model only (no observation update). Used to weight a
        hypothesis mixture for the *next* step's likelihood, rather than
        using an unpropagated posterior as if it were that step's prior."""
        p = self.responding_state[0]
        return p * (1 - self.epsilon_c) + (1 - p) * self.epsilon_n
