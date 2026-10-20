"""Append/refresh the isolated note from saved nonlinear results."""
import numpy as np
import pandas as pd
import nonlinear_collective_analysis as a

def main():
    selected=pd.read_csv(a.OUT/'selected_results.csv');bound=pd.read_csv(a.OUT/'boundary_estimates.csv');allrows=pd.read_csv(a.OUT/'summary_with_intervals.csv')
    warm=bound.query('temperature_K==300').iloc[0];cold=bound.query('temperature_K==.1').iloc[0]
    text=r'''
\clearpage
\section*{Nonlinear along-trap check}

The growing fluctuations were next tested with the actual nonlinear trap and
pair forces. A restriction to 40 coordinates repeats a two-column cell along
$x$ while retaining all five cells along $y$, including the fastest linear
modes. Each restricted coordinate represents five particles moving together,
so projected diffusion is $D/5$. No vertical switches were seen in the initial
64-run warm $\tau=8$ s check. Because this restriction removes spatial modes
and individual thermal noise, it was not used to estimate the line.

The unrestricted reduced model retains one signed coordinate $s_i$ per trap:
\begin{equation}
 ds_i=\frac{F_i^{\rm mag}(\boldsymbol s,\theta)+F_{\rm trap}(s_i)}{\gamma}\,dt
       +\sqrt{2D}\,dW_i,\qquad D=\frac{k_BT}{\gamma}.
\end{equation}
There are 200 coordinates with independent particle noise. The dipolar force
uses the audited finite-cutoff planar images; no fitted rate, force, or noise
factor is introduced. Horizontals and verticals both evolve, rather than
imposing a background trajectory. Transverse and out-of-plane motion are
omitted. This is numerical sampling of a reduced overdamped model, not a new
LAMMPS campaign or a closed-form transition law.

Each step takes explicit magnetic drift and a Brownian increment, then solves
the scalar implicit trap update. This retains the shallow interior, cusp, and
stiff outer branch. All runs start from the same ordered configuration,
$s_i=5\,\mu\mathrm m$. Final occupation $s_i<0$ is measured at
$t_{\rm obs}=\tau-0.05$ s, matching the saved 20 Hz observations. First-ever
crossings are also recorded, so returns are allowed rather than assumed absent.

The selected warm results are:
\begin{center}
\begin{tabular}{c c c c}
$\tau$ [s] & model $p_v$ & sampling SE & measured $p_v$\\
'''
    measured={7:.001,8:.009,9:.017}
    for tau in [7,8,9]:
        r=selected.query('temperature_K==300 and tau_s==@tau').iloc[0]
        text+=f'{tau} & {r.p_vertical_final:.5f} & {r.standard_error_over_independent_runs:.5f} & {measured[tau]:.3f}'+r'\\'+'\n'
    text+=r'''\end{tabular}
\end{center}
The $8$ s model mean is based on 256 independent runs; the neighbouring
protocols use 128 each. Standard errors are computed across runs, preserving
particle dependence within a run. These model means overpredict the sampled
warm final fractions somewhat, but reproduce persistent occupation and the
switching interval, unlike the linear tail's artificial collapse.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.65\linewidth]{nonlinear_retained_switches.pdf}
\caption{Warm final-well occupation as the field rotates. The nonlinear model
retains vertical switches after the unstable interval. Shading shows approximate
$95\%$ sampling bands from independent model realizations; it excludes model
and integration bias. Existing data are shown as points.}
\end{figure}

For the cold protocols, no crossings were seen at $30$ s in 64 model runs.
This does not imply an exactly zero probability. At $40$ s the selected model
fraction is COLDPROB with standard error COLDSE, versus $0.058$ in the data.
The selected $40$ s result uses a $0.0025$ s integration step.

Linear interpolation of the sampled \emph{final} crossing fraction to $1\%$
gives warm $\omega_{\rm L}\simeq WARMROOT\,\mathrm{rad/s}$ and cold
$\omega_{\rm L}\simeq COLDROOT\,\mathrm{rad/s}$. The measured estimates are
$0.19362$ and $0.05010\,\mathrm{rad/s}$. The warm model estimate is about
$5\%$ higher. Sampled frequency brackets are $0.19635$--$0.22440$ for warm
and $0.03927$--$0.05236\,\mathrm{rad/s}$ for cold. These are grid brackets,
not confidence intervals, and interpolation need not capture curvature
between the protocols. They are appended to the separate theoretical table,
retaining all earlier failed approximations and the empirical table.

Bootstrap probability intervals resample independent model runs. At a
zero-event endpoint a conservative one-sided run-level bound is saved instead
of a degenerate zero-width interval. The additional warm root bootstrap
interval, conditional on retaining the sampled bracket, excludes discretization
and model bias. Shared random seeds across protocols are handled by paired
run-index resampling. No cold root confidence interval is assigned because
its fast endpoint has zero observed crossings.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.65\linewidth]{nonlinear_boundary_comparison.pdf}
\caption{Final crossing fractions versus rotation frequency in the nonlinear
along-trap model and existing data. Model bars show bootstrap sampling intervals;
the zero-event cold endpoint has a conservative run-level upper bound.
The dashed line is the $1\%$ criterion. Both temperature scales are recovered
approximately, with a remaining warm bias.}
\end{figure}

The nonlinear magnetic evaluator agrees with the previous direct per-site
force calculation to $1.3\times10^{-14}\,\mathrm{pN}$. Zero-temperature
trajectories converge toward the coherent Radau solution as the step is
halved. At warm $\tau=8$ s, steps $0.005$, $0.0025$, and $0.00125$ s give
final fractions $0.01215$, $0.01176$, and $0.01055$. Their sampling standard
errors are $0.00067$, $0.00066$, and $0.00092$, respectively. The finest
check uses an independent seed and 128 runs; these comparisons are compatible
with sampling uncertainty but do not establish arbitrarily precise
step convergence. The cold $40$ s coarse and refined results are COLDCOARSE
and COLDPROB, with sampling uncertainty saved alongside them.

\begin{samepage}
The nonlinear model supports collective fluctuation growth followed by
persistent switching as a plausible explanation of the frequency scale.
It remains a reduced-coordinate prediction, with omitted transverse motion,
sampling uncertainty, and possible numerical or model bias. A compact paper
statement can use this as a mechanism check; precision transition lines or
a closed-form temperature law would require further work.
\end{samepage}
'''
    coldrow=selected.query('temperature_K==.1 and tau_s==40').iloc[0]
    coarserow=allrows.query('temperature_K==.1 and tau_s==40 and dt_requested_s==.005').iloc[0]
    for name,value in [('COLDPROB',f'${coldrow.p_vertical_final:.5f}$'),('COLDSE',f'${coldrow.standard_error_over_independent_runs:.5f}$'),
                       ('COLDCOARSE',f'${coarserow.p_vertical_final:.5f}$'),('WARMROOT',f'{warm.omega_estimate_rad_s:.5f}'),('COLDROOT',f'{cold.omega_estimate_rad_s:.5f}')]:text=text.replace(name,value)
    note=a.p.REPO.parent/'latex/transition-timescale-note.tex';old=note.read_text();marker=r'\section*{Nonlinear along-trap check}'
    if marker in old:old=old[:old.index(marker)]+r'\end{document}'+'\n'
    old=old.replace('{../repo/data/figures/collective-stability/}}','{../repo/data/figures/collective-stability/}{repo/data/figures/nonlinear-collective/}{../repo/data/figures/nonlinear-collective/}}')
    note.write_text(old.replace(r'\end{document}',text+'\n'+r'\end{document}'))
if __name__=='__main__':main()
