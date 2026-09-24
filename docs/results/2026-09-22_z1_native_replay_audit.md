# Native replay audit — strict failure, bounded observed differences

Status: original strict fidelity criterion failed; fresh engineering-equivalence validation passed. The contact timestep comparison is still pending. No production gains, safety caps, rewards or training changed.

## Evidence

Native source capture45736772 completed allfourarms. Strict replay45736947 stopped allfour before finersteps. Complete native-only audit45737418 then finished allfour, retaining exit2/failedstrictverdict and fullprefixdata. Failedoptional-sensor preflight45736101 is separate: actualtask lacks optional pointgeometrysensor, and reducer-aware validity correction was independentlyreviewed before successfulcapture.

Each arm has128worlds at externalgoal2.2m/s. Arms0/2 originalcontroller,1/3 fixedwrist128/25.6; doses.2/.3 respectively. Otherfivejointsgains remain frompolicy. Common horizon0.28s, fixedsourcecommands/gains, terminalcommandholdexplicitlycounterfactual. Prefix comparisons stop at eachworld's firstnative terminalinterval; no next-episode rows used.

|Arm|Max reaction error / own cap|Max arm-speed error / cap|Max arm-angle error rad|Max delivered error Ns|
|---|---:|---:|---:|---:|
|0 originaldose.2|0.047114%|0.030547%|0.000008452|0.000149388|
|1 candidatedose.2|0.173771%|0.213177%|0.000083994|0.000165761|
|2 originaldose.3|0.021702%|0.024969%|0.000008035|0.000077516|
|3 candidatedose.3|0.063915%|0.018598%|0.000007084|0.000112295|

Independently recomputed from rawtraces: exact target/ctrl/encoderbias, exact contact/finalization/reason, zero perworld/perjoint peak-capclassification flips acrosseligibleprefix. Appliedgainidentity checked duringreplay; fullinitialstate/modelrestore passed. No nonfinite or monitoredcapacityoverflow reported. Optionalpoint-slotgeometryquality unavailable, notmeasuredzero.

Firststrictfailures occur beforeanysourceworldreset, so laterotherworldautoresets cannotaloneexplain firstdivergence. No directphysicalwrite found in omittedfullstepreward/termination/metric callbacks during targetedreview; bookkeeping/EMA/payout differs. Underlyingcause remainsunidentified. Do not claim provedGPU nondeterminism or exactreproduction.

The originalcriterion (qposatol1e-6,qvel1e-4,reaction/delivered1e-6,rtol1e-5) remainsfailed. Nativeclosedlooprepeat variationsprovidecontext butarenot fullprefixfixed-command errorbounds.

## Prospective next check

After independentrawreview, an explicit engineeringerrorbudget was frozen BEFORE freshvalidationcapture45737719: maximum0.5%ofeachreaction/armvelocitycap; hingeposition0.0002rad; slideposition0.0001m; nonarmhinge/slidevelocity0.005inrespectiveunits; delivered0.001Ns; exactcommands/discreteevents, nooverflow and nopeak-capclassificationflips. These are diagnosticresolutionbudgets, notchangedrobotcaps or empiricalsafeactuatorranges. Reaction/velocitybudget isonetenththestated5%-of-cap refinementscreen. Fullprotocol retainedin `.superpowers/sdd/2026-09-22-wrist-calibration/engineering-fidelity-protocol.md`.

Freshallfourvalidationmustpassbeforefiner1/.5mslevels areauthorized. Itsfailurecannotbequietlyfixedbylooseningbudgets. Originalstrictfailurestaysvisible alongsideanyengineeringacceptance. No numericalconvergence, hardwarequalification ortrainingendorsement followsfromthisaudit alone.

Rawdata: [audit artifacts](assets/2026-09-22_wrist_calibration/native-audit/); [source captures](assets/2026-09-22_wrist_calibration/contact-attempt2/). Independent interpretation and methodnotes retained in the experiment workspace.

## Fresh validation outcome

Fresh source captures45737719 completed successfully after the engineering protocol was frozen. Validation45737812 completed all four native-only audits, retaining strict failures and exit code2. The separate engineering-equivalence results passed in every arm, with exact commands and discrete events, zero prefix peak-cap classification flips, and no nonfinite or monitored overflow failures.

The largest fresh discrepancies were0.1526% of a joint impulse cap,0.1322% of the arm-speed cap,0.000042053rad arm angle,0.000010088m nail displacement and0.000167474N·s delivered impulse. These are below the prospectively chosen budgets. The common horizon remains0.28s. This permits only a bounded engineering-resolution comparison at1/.5ms; it does not turn the original strict failures into passes or establish their cause.

[Fresh validation artifacts](assets/2026-09-22_wrist_calibration/engineering-validation/) preserve both verdicts and the complete paired data.
