import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# Streamlit setup
# ============================================================

st.set_page_config(
    page_title="Stochastic de novo beneficial mutation model",
    layout="wide",
)

st.title("Stochastic phase 1 + deterministic phase 2 for a de novo beneficial mutation")

st.markdown(
    r"""
This app models a beneficial mutation that is initially absent.

It uses:

\[
P_{\text{est}}
=
\frac{1-e^{-2s}}{1-e^{-4N_es}}
\]

Then it draws approximately:

\[
\frac{1}{P_{\text{est}}}
\]

independent de novo mutant lineages: all but one go extinct, and one reaches the establishment copy count.

The successful lineage is horizontally shifted so that it reaches establishment at the expected establishment time.
"""
)


# ============================================================
# Sidebar controls
# ============================================================

with st.sidebar:
    st.header("Parameters")

    Ne = st.number_input(
        "Effective population size, Ne",
        min_value=10.0,
        max_value=1e12,
        value=50_000.0,
        step=1_000.0,
        format="%.8g",
    )

    s = st.number_input(
        "Selection coefficient, s",
        min_value=1e-8,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.8g",
    )

    mu = st.number_input(
        "Beneficial mutation rate per allele copy per generation, μ",
        min_value=1e-15,
        max_value=1e-2,
        value=1e-6,
        step=1e-7,
        format="%.10g",
    )

    target_p = st.number_input(
        "Target allele frequency",
        min_value=1e-12,
        max_value=0.999999,
        value=0.5,
        step=0.01,
        format="%.8g",
    )

    max_lineages = st.slider(
        "Maximum stochastic lineages to plot",
        min_value=2,
        max_value=1000,
        value=100,
        step=1,
    )

    max_attempts = st.number_input(
        "Maximum rejection-sampling attempts",
        min_value=100,
        max_value=2_000_000,
        value=100_000,
        step=1_000,
    )

    max_phase1_generations = st.number_input(
        "Maximum generations per stochastic phase-1 path",
        min_value=10,
        max_value=1_000_000,
        value=50_000,
        step=100,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=1_000_000,
        value=7,
        step=1,
    )

    y_scale = st.selectbox("Y-axis scale", ["log", "linear"])

    auto_x_axis = st.checkbox("Auto x-axis range", value=True)

    manual_max_generations = st.number_input(
        "Manual max generations",
        min_value=10.0,
        max_value=1e9,
        value=10_000.0,
        step=100.0,
        format="%.8g",
    )


# ============================================================
# Probability and model formulas
# ============================================================

def exact_establishment_probability(Ne: float, s: float) -> float:
    """
    Exact diffusion approximation for the fixation / establishment probability
    of one new beneficial allele:

        P_est = (1 - exp(-2s)) / (1 - exp(-4Ne s))

    Uses expm1 for numerical stability:

        1 - exp(-x) = -expm1(-x)

    Neutral fallback as s -> 0:

        P_fix = 1 / (2Ne)
    """
    numerator = -np.expm1(-2.0 * s)
    denominator = -np.expm1(-4.0 * Ne * s)

    if np.isclose(denominator, 0.0):
        return 1.0 / (2.0 * Ne)

    return numerator / denominator


P_est = exact_establishment_probability(Ne, s)

initial_copies = 1
p_initial = 1.0 / (2.0 * Ne)

# Establishment threshold in allele copies.
#
# Rule of thumb:
#
#   X_est ≈ 1/s
#
raw_establishment_copies = int(np.ceil(1.0 / s))
max_possible_copies = int(np.floor(2.0 * Ne))

if raw_establishment_copies > max_possible_copies:
    st.warning(
        "The threshold 1/s exceeds total allele copies 2Ne. "
        "The establishment copy threshold is being clamped to 2Ne."
    )

establishment_copies = min(raw_establishment_copies, max_possible_copies)
p_est = establishment_copies / (2.0 * Ne)

# Mutation arrival.
#
# In a diploid population, there are about 2Ne allele copies.
#
# New beneficial mutant copies per generation:
#
#   2Ne μ
#
mutation_arrival_rate = 2.0 * Ne * mu

# Successful mutant arrival rate:
#
#   2Ne μ P_est
#
successful_arrival_rate = mutation_arrival_rate * P_est

if successful_arrival_rate > 0:
    t_wait_success = 1.0 / successful_arrival_rate
else:
    t_wait_success = np.inf

# Expected number of de novo mutant lineages per successful establishment.
expected_lineages_per_success = 1.0 / P_est

requested_lineages = int(np.ceil(expected_lineages_per_success))
n_lineages = min(requested_lineages, max_lineages)

n_successful = 1
n_extinct = max(n_lineages - 1, 0)

# Expected time from one mutant copy to establishment under exponential expectation:
#
#   E[X_t] = exp(st)
#
# Establishment occurs in expectation when:
#
#   exp(st) = X_est
#
# so:
#
#   t = log(X_est)/s
#
if establishment_copies > 1:
    t_phase1_expectation = np.log(establishment_copies) / s
else:
    t_phase1_expectation = 0.0

# Expected absolute establishment time:
#
#   waiting time for successful mutation + expected stochastic phase-1 time
#
t_establishment_expected_absolute = t_wait_success + t_phase1_expectation

# Deterministic phase-2 time from p_est to target_p:
#
#   dp/dt = s p(1-p)
#
#   t = (1/s) log( p1(1-p0) / (p0(1-p1)) )
#
if 0 < p_est < target_p < 1:
    t_phase2_to_target = (1.0 / s) * np.log(
        (target_p * (1.0 - p_est)) / (p_est * (1.0 - target_p))
    )
else:
    t_phase2_to_target = np.nan

if np.isfinite(t_phase2_to_target):
    t_target_absolute = t_establishment_expected_absolute + t_phase2_to_target
else:
    t_target_absolute = np.nan


# ============================================================
# Stochastic branching-process simulation
# ============================================================

def simulate_branching_path(
    s: float,
    establishment_copies: int,
    rng: np.random.Generator,
    max_generations: int,
):
    """
    Simulate a rare beneficial mutant lineage with a branching process:

        X_{t+1} ~ Poisson((1+s) X_t)

    Stop conditions:
    - extinct: X_t = 0
    - established: X_t >= establishment_copies
    - censored: max_generations reached

    Returns:
    - path: array of copy counts
    - status: "extinct", "established", or "censored"
    """
    copies = 1
    path = [copies]

    for _ in range(max_generations):
        lam = (1.0 + s) * copies
        copies = int(rng.poisson(lam))

        if copies >= establishment_copies:
            path.append(establishment_copies)
            return np.array(path, dtype=float), "established"

        path.append(copies)

        if copies <= 0:
            return np.array(path, dtype=float), "extinct"

    return np.array(path, dtype=float), "censored"


def sample_conditioned_path(
    desired_status: str,
    s: float,
    establishment_copies: int,
    rng: np.random.Generator,
    max_generations: int,
    max_attempts: int,
):
    """
    Rejection-sample a path with a desired outcome:
    - desired_status = "extinct"
    - desired_status = "established"
    """
    for attempt in range(int(max_attempts)):
        path, status = simulate_branching_path(
            s=s,
            establishment_copies=establishment_copies,
            rng=rng,
            max_generations=int(max_generations),
        )

        if status == desired_status:
            return path, attempt + 1

    return None, int(max_attempts)


rng = np.random.default_rng(int(seed))


# ============================================================
# Sample extinct paths
# ============================================================

extinct_paths = []
extinct_birth_times = []
extinct_attempt_counts = []

for _ in range(n_extinct):
    path, attempts_used = sample_conditioned_path(
        desired_status="extinct",
        s=s,
        establishment_copies=establishment_copies,
        rng=rng,
        max_generations=int(max_phase1_generations),
        max_attempts=int(max_attempts),
    )

    if path is not None:
        extinct_paths.append(path)
        extinct_attempt_counts.append(attempts_used)

        # Failed mutations are placed before the expected successful-establishment event.
        #
        # Conditional on their number in a time interval, Poisson arrivals are
        # uniformly distributed across that interval.
        if np.isfinite(t_wait_success):
            birth_time = rng.uniform(0.0, t_wait_success)
        else:
            birth_time = 0.0

        extinct_birth_times.append(birth_time)


# ============================================================
# Sample one successful path
# ============================================================

successful_path, success_attempts_used = sample_conditioned_path(
    desired_status="established",
    s=s,
    establishment_copies=establishment_copies,
    rng=rng,
    max_generations=int(max_phase1_generations),
    max_attempts=int(max_attempts),
)

if successful_path is None:
    st.error(
        "Could not rejection-sample a successful establishing trajectory. "
        "Increase selection s, increase max rejection-sampling attempts, "
        "or increase max generations per stochastic path."
    )
    st.stop()

successful_path_duration = len(successful_path) - 1

# Shift the successful path so it reaches establishment at the expected
# absolute establishment time.
successful_birth_time = t_establishment_expected_absolute - successful_path_duration

# If the sampled path is longer than the expected placement would allow,
# clamp it to start at zero.
if successful_birth_time < 0:
    successful_birth_time = 0.0

successful_establishment_time = successful_birth_time + successful_path_duration


# ============================================================
# Deterministic phase 2
# ============================================================

def logistic_phase2(t, t_start, p_start, s):
    """
    Deterministic selection after establishment:

        dp/dt = s p(1-p)

    Solution:

        p(t) = 1 / [1 + ((1-p_start)/p_start) exp(-s(t-t_start))]
    """
    tau = t - t_start
    return 1.0 / (
        1.0 + ((1.0 - p_start) / p_start) * np.exp(-s * tau)
    )


if np.isfinite(t_phase2_to_target):
    realized_target_time = successful_establishment_time + t_phase2_to_target
else:
    realized_target_time = np.nan

if auto_x_axis:
    candidate_times = [
        t_wait_success,
        t_establishment_expected_absolute,
        successful_establishment_time,
        realized_target_time,
    ]
    finite_times = [x for x in candidate_times if np.isfinite(x) and x > 0]

    if finite_times:
        plot_max_t = max(100.0, max(finite_times) * 1.05)
    else:
        plot_max_t = 1000.0
else:
    plot_max_t = manual_max_generations


# ============================================================
# Plot helpers
# ============================================================

def copy_path_to_frequency(copy_path, Ne):
    return np.asarray(copy_path, dtype=float) / (2.0 * Ne)


def plot_copy_path(
    ax,
    birth_time,
    copy_path,
    Ne,
    label=None,
    linewidth=1.0,
    alpha=0.4,
    linestyle="-",
):
    generations = birth_time + np.arange(len(copy_path))
    frequencies = copy_path_to_frequency(copy_path, Ne)

    if y_scale == "log":
        # Zero cannot be shown on a log axis.
        frequencies = np.where(frequencies > 0, frequencies, np.nan)

    ax.plot(
        generations,
        frequencies,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        label=label,
    )


# ============================================================
# Main plot
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Extinct stochastic lineages.
for i, (birth_time, path) in enumerate(zip(extinct_birth_times, extinct_paths)):
    label = "Extinct stochastic lineages" if i == 0 else None
    plot_copy_path(
        ax=ax,
        birth_time=birth_time,
        copy_path=path,
        Ne=Ne,
        label=label,
        linewidth=0.8,
        alpha=0.25,
    )

# One successful stochastic lineage.
plot_copy_path(
    ax=ax,
    birth_time=successful_birth_time,
    copy_path=successful_path,
    Ne=Ne,
    label="One stochastic lineage reaches establishment",
    linewidth=2.8,
    alpha=1.0,
)

# Dotted expectation line for phase 1.
#
# Draw it from the expected birth of the successful mutant to expected establishment.
#
expected_successful_birth_time = t_wait_success

expected_line_t = np.linspace(
    expected_successful_birth_time,
    expected_successful_birth_time + t_phase1_expectation,
    400,
)

expected_copies = np.exp(s * (expected_line_t - expected_successful_birth_time))
expected_copies = np.minimum(expected_copies, establishment_copies)
expected_p = expected_copies / (2.0 * Ne)

ax.plot(
    expected_line_t,
    expected_p,
    linestyle=":",
    linewidth=2.2,
    label=r"Phase-1 expectation: $E[X_t]=e^{st}$",
)

# Horizontal establishment threshold.
ax.axhline(
    p_est,
    linestyle="--",
    linewidth=1.6,
    label=f"Establishment threshold ≈ {establishment_copies:,} copies",
)

# Deterministic phase 2 from the realized establishment time.
if successful_establishment_time < plot_max_t:
    t2 = np.linspace(successful_establishment_time, plot_max_t, 1000)
    p2 = logistic_phase2(
        t=t2,
        t_start=successful_establishment_time,
        p_start=p_est,
        s=s,
    )

    ax.plot(
        t2,
        p2,
        linewidth=2.8,
        label="Phase 2: deterministic selection",
    )

# Vertical reference lines.
if np.isfinite(t_wait_success) and t_wait_success <= plot_max_t:
    ax.axvline(
        t_wait_success,
        linestyle="--",
        linewidth=1,
        label="Expected successful mutation birth",
    )

if np.isfinite(t_establishment_expected_absolute) and t_establishment_expected_absolute <= plot_max_t:
    ax.axvline(
        t_establishment_expected_absolute,
        linestyle=":",
        linewidth=1.4,
        label="Expected establishment time",
    )

if np.isfinite(successful_establishment_time) and successful_establishment_time <= plot_max_t:
    ax.axvline(
        successful_establishment_time,
        linestyle="--",
        linewidth=1,
        label="Successful path reaches establishment",
    )

if np.isfinite(realized_target_time) and realized_target_time <= plot_max_t:
    ax.axvline(
        realized_target_time,
        linestyle=":",
        linewidth=1,
        label=f"Target frequency p = {target_p:g}",
    )

ax.set_xlim(0.0, plot_max_t)

if y_scale == "log":
    ax.set_yscale("log")
    ymin = max(1.0 / (20.0 * Ne), 1e-15)
    ax.set_ylim(ymin, 1.05)
else:
    ax.set_ylim(0.0, 1.05)

ax.set_xlabel("Generation, t")
ax.set_ylabel("Allele frequency, p")
ax.set_title("Stochastic phase 1 with exact P_est and deterministic phase 2")
ax.grid(True, which="both", alpha=0.3)


# Secondary y-axis: mutant allele copies.
def freq_to_copies(p):
    return 2.0 * Ne * p


def copies_to_freq(copies):
    return copies / (2.0 * Ne)


secax = ax.secondary_yaxis("right", functions=(freq_to_copies, copies_to_freq))
secax.set_ylabel("Mutant allele copies")

ax.legend(loc="best", fontsize=8)

st.pyplot(fig)


# ============================================================
# Summary table
# ============================================================

st.subheader("Computed quantities")

summary = [
    ("Effective population size, Ne", Ne),
    ("Selection coefficient, s", s),
    ("Mutation rate per allele copy per generation, μ", mu),
    ("Exact P_est", P_est),
    ("Expected lineages per successful establishment, 1/P_est", expected_lineages_per_success),
    ("Requested stochastic lineages, ceil(1/P_est)", requested_lineages),
    ("Stochastic lineages plotted", n_lineages),
    ("Extinct stochastic lineages plotted", len(extinct_paths)),
    ("Successful stochastic lineages plotted", 1),
    ("Initial mutant copy count", initial_copies),
    ("Initial allele frequency", p_initial),
    ("Establishment copy count ≈ 1/s", establishment_copies),
    ("Establishment allele frequency", p_est),
    ("New beneficial mutant copies per generation, 2Neμ", mutation_arrival_rate),
    ("Successful mutant arrival rate, 2NeμP_est", successful_arrival_rate),
    ("Expected waiting time for successful mutation", t_wait_success),
    ("Expected phase-1 time from one copy to establishment", t_phase1_expectation),
    ("Expected absolute establishment time", t_establishment_expected_absolute),
    ("Sampled successful phase-1 duration", successful_path_duration),
    ("Shifted successful mutant birth time", successful_birth_time),
    ("Shifted successful establishment time", successful_establishment_time),
    (f"Phase-2 time from establishment to p = {target_p:g}", t_phase2_to_target),
    (f"Total shifted time to p = {target_p:g}", realized_target_time),
    ("Attempts used to sample successful path", success_attempts_used),
]

st.dataframe(
    [{"quantity": k, "value": v} for k, v in summary],
    use_container_width=True,
)


# ============================================================
# Equations
# ============================================================

st.subheader("Model equations")

st.markdown(
    r"""
One new mutant allele copy begins at frequency:

\[
p_0 = \frac{1}{2N_e}
\]

The exact establishment/fixation approximation used here is:

\[
P_{\text{est}}
=
\frac{1-e^{-2s}}{1-e^{-4N_es}}
\]

The expected number of new mutant lineages per successful establishment is:

\[
\frac{1}{P_{\text{est}}}
\]

The rate of new beneficial mutant copies is:

\[
2N_e\mu
\]

The successful-mutant arrival rate is:

\[
2N_e\mu P_{\text{est}}
\]

So the expected waiting time for a successful de novo mutation is:

\[
T_{\text{wait}}
=
\frac{1}{2N_e\mu P_{\text{est}}}
\]

The stochastic phase-1 simulation uses:

\[
X_{t+1} \sim \operatorname{Poisson}((1+s)X_t)
\]

The establishment copy threshold is:

\[
X_{\text{est}} \approx \frac{1}{s}
\]

The phase-1 expectation is:

\[
\mathbb{E}[X_t] = e^{st}
\]

After establishment, deterministic selection is modeled by:

\[
\frac{dp}{dt} = sp(1-p)
\]

with solution:

\[
p_t =
\frac{1}{
1+
\left(\frac{1-p_{\text{est}}}{p_{\text{est}}}\right)e^{-st}
}
\]
"""
)

st.warning(
    """
This is a stylized model. The early phase is simulated as a branching process, not a full Wright-Fisher population.
The model conditions the displayed sample on exactly one lineage establishing and the rest going extinct.
When P_est is very small, exact sampling can be slow; increase max attempts or reduce the number of plotted lineages.
"""
)