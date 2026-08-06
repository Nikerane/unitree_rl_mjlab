# Khadiv FIC trajectory-tracking-term audit

## Verdict

**Yes—within the paper's hopper / sim-to-real experiment, the fixed-gain PD
controller (FIC) was explicitly evaluated both without and with the same
one-step trajectory-tracking reward.** This is not an extrapolation from VIC:
the paper's Figure 5 has panels for “Fixed gain control without trajectory
tracking reward term” and “Fixed gain control with trajectory tracking reward
term,” and the text says the authors repeated training with the penalty enabled
for **both fixed and variable gain policies**. The fixed-gain output changed
little, while the VIC term resolves an additional desired-position/gain
non-identifiability.

The answer is therefore **not** “RTT is VIC-only.” It is equally important not
to overstate it: FIC was also studied *without* RTT; the paper's initial
controller comparison is not described as carrying RTT; and the later
fixed-base KUKA simulation specifies a separate task reward but does not state
that RTT was added there.

## Exact paper and terminology

Miroslav Bogdanovic, Majid Khadiv, and Ludovic Righetti, *Learning Variable
Impedance Control for Contact Sensitive Tasks*, IEEE Robotics and Automation
Letters 5(4), 6129–6136 (2020), DOI
[10.1109/LRA.2020.3011379](https://doi.org/10.1109/LRA.2020.3011379), author
preprint [arXiv:1907.07500v2](https://arxiv.org/abs/1907.07500).

The paper calls the comparator **“fixed gain PD control”** (not FIC): its
policy outputs desired joint positions and uses predefined gains,
`tau = Kp (q_des - q) - Kd qdot` (Section II, Eq. (2), p. 6130 / PDF p. 2).
Its variable-gain policy outputs both desired positions and gains (Eq. (3)).
Thus “fixed impedance” here means fixed PD gains while retaining policy-chosen
joint position targets.

The source notation is already `r_tt`; it is not merely a name introduced by
this repository. In Section IV-C (“Trajectory tracking reward term”, p. 6133
/ PDF p. 5), the paper defines

```text
r_tt = -k || q_des^t - q^{t+1} ||^2 .
```

This is the same causal one-control-step structure being reproduced for the Z1:
compare the target issued at `t` against the state reached at `t+1`, rather than
penalizing movement away from the current state. The paper explains that a
target reached next step receives zero tracking penalty.

## Direct evidence on FIC

1. **Definition of the FIC arm.** Section II Eq. (2) defines fixed-gain PD as
   policy-chosen `q_des` with fixed `Kp, Kd`. That makes the `r_tt` expression
   well-defined for FIC as well as VIC. [Primary PDF, Section II,
   p. 6130](https://arxiv.org/pdf/1907.07500).

2. **Figure-level ablation.** Figure 5 (p. 6133) labels panels (c) and (d) as
   fixed-gain control *without* and *with* the trajectory-tracking term.
   [Primary PDF, Fig. 5](https://arxiv.org/pdf/1907.07500).

3. **Text-level confirmation.** Immediately after the formula, the authors say
   they repeat the preceding training with the penalty enabled for “both fixed
   and variable gain policies,” and report similar best task performance for
   both. The next paragraph reports that the fixed-gain output “practically
   does not change,” whereas the VIC desired positions become more closely
   trackable. [Primary PDF, Section IV-C, pp. 6133–6134](https://arxiv.org/pdf/1907.07500).

## What is shared, and what is VIC-specific

| Question | Paper-supported answer |
|---|---|
| Do FIC and VIC share a task objective? | In the hopper RTT ablation, yes: the same added `r_tt` is enabled for both. The base hopper reward is also shared across controller parameterizations. |
| Does FIC retain `r_tt` while gains remain fixed? | Yes. Fig. 5(c,d) and the accompanying text are direct evidence. |
| Is the main rationale equally strong for FIC? | No. The paper identifies VIC's extra ambiguity: a distant `q_des` plus a smaller gain can yield the same torque. It says FIC cannot vary gains and its learned position outputs changed little with RTT. |
| Is RTT claimed for every paper experiment? | No. The paper introduces it in the hopper sim-to-real section. The fixed-base KUKA section instead enumerates five circle/contact reward components and does not say RTT is added. |

## Consequence for the Z1 Stage-1 design

A fixed-gain Z1 arm with this term is paper-aligned **as an FIC+RTT treatment**,
but it should not be described as the paper's only FIC baseline. The rigorous
comparison is FIC without RTT versus FIC with RTT (and later VIC with the same
term), holding task reward and action semantics fixed. If resource limits make
the no-RTT arm an engineering-only control in Stage 1, label the trained result
as a *joint-position + RTT bundle*, not a causal estimate of the action-space
change or of RTT alone.

## Scope and source check

This audit used the authors' open preprint (v2, 14 July 2020), whose title,
authors, and journal version are also listed by the [TUM publication
record](https://portal.fis.tum.de/en/publications/learning-variable-impedance-control-for-contact-sensitive-tasks/).
The first author's [publication page](https://miroslavbogdanovic.github.io/)
links to the same arXiv paper and video, but no official implementation source
is linked there; the conclusion above rests on the paper itself, not a
third-party code summary.
