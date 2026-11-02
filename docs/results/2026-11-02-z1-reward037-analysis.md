# Z1 reward037: all-on rewards and height pilots

The combined reward recipe is the strongest nominal baseline observed so far: all six main policies reached native30mm in32/32 matched contexts at both100 and500 updates, with zero recorded whole-attempt velocity/load and contact-window velocity/impulse violations. Both initial exploration settings repeat this across three seeds. This supports keeping all three selected task rewards ON; it does not establish that each term is necessary.

## Material Passport

Verification status: ANALYZED (saved-evidence analysis, no physical rerun). Inputs: eight checksum-verified local archives,16cases/512nominal starts; full training/raw trees independently backed up on the verified SSD. Two independent count derivations plus GPT-6 Astra Ultra review. Statistical unit: trained seed, three per base setting; repeated32contexts and nested checkpoints are not independent training replications. Height pilots have one seed per setting and nominal evaluation only. User requested actual Fable/Opus/Kimi3.1 plus Ultra review; raw model responses and correction rounds are preserved in the evidence panel directory. Model agreement is review advice, not experimental evidence.

## Full seed/checkpoint results

All rows retain split50/200ms payout,event-aware critic43actor/47critic,ordinary learned state-independent Gaussian,variable impedance,existing constraints and200ms aftermath. Added impulse/depth-progress/once-completion rewards are ON together. Legacy repeating completion remains OFF. Std is the INITIAL position std, not a frozen training std. Each count is out of32.

| Setting | Seed | Updates | Productive | Native30mm | Whole velocity | Whole load | Contact velocity | Contact impulse | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| std0.3 | 2026110201 | 100 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.3 | 2026110201 | 500 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.3 | 2026110202 | 100 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.3 | 2026110202 | 500 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.3 | 2026110203 | 100 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.3 | 2026110203 | 500 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.1 | 2026110201 | 100 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.1 | 2026110201 | 500 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.1 | 2026110202 | 100 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.1 | 2026110202 | 500 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.1 | 2026110203 | 100 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.1 | 2026110203 | 500 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.3 + heightDR | 2026110201 | 100 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.3 + heightDR | 2026110201 | 500 | 32 | 32 | 0 | 1 | 0 | 0 | 0 |
| std0.1 + heightDR | 2026110201 | 100 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |
| std0.1 + heightDR | 2026110201 | 500 | 32 | 32 | 0 | 0 | 0 | 0 | 0 |

Every contact window was observed and complete:801 onset-inclusive samples through200ms. Native peak-depth median is32mm in every case, a simulator ceiling, not evidence of maintained depth or robustness.

## What changed versus rewards OFF

| Setting | 036 OFF native30mm at100 | 037 ON at100 | 036 OFF at500 | 037 ON at500 |
|---|---|---|---|---|
| std0.3, three seeds | 0 / 0 / 0 | 32 / 32 / 32 | 0 / 30 / 0 | 32 / 32 / 32 |
| std0.1, three seeds | 0 / 0 / 0 | 32 / 32 / 32 | 0 / 0 / 2 | 32 / 32 / 32 |

Each vector entry is out of32; seeds2026110201/02/03. Productive strikes were already32/32 in036, so the advance is reliable full depth, rather than newly making contact. The036std0.3 early whole-load/contact-impulse vector3/0/22 is now0/0/0; at500botholdsplitsettingsalreadyrecordedzero violations.

Independent saved-initialization comparison found ALL SIX037/036 pairs equal in full model/normalizer/RNG fingerprints, initial actor/shared43critic input hashes,std and seeds. This strengthens the bundle comparison beyond merely sharing seed labels. It is not a full simulator-state equivalence proof or a formal population causal estimate; only three seeds per setting. Since the three rewards changed together, this cannot attribute the improvement to impulse alone, progress alone, or completion alone.

## What later training changed

Depth and recorded base violations were already saturated by100updates and stayed there at500. Canonical delivered impulse increased in all three std0.3 seeds, two of three std0.1 seeds and both height pilots. Std0.1 seed02 decreased slightly,0.256595→0.252997N·s. No full-depth/productivity regression occurred. The DRstd0.3 policy developed one whole-attempt load violation by500.

| Setting | Canonical delivered impulse100→500, N·s by seed |
|---|---|
| std0.3 | 0.2864→0.3176; 0.2418→0.2938; 0.2473→0.3112 |
| std0.1 | 0.2730→0.3338; 0.2566→0.2530; 0.2995→0.3318 |
| std0.3 + heightDR | 0.3007→0.3143 |
| std0.1 + heightDR | 0.2977→0.3305 |

At500updates the equal-seed averages are0.3075N·s(std0.3) and0.3062N·s(std0.1), with overlapping per-seed ranges. There is no justified overall winner. This study did NOT test gSDE or observation-dependent std.

## The height-pilot exception

HeightDRstd0.3,seed01,500updates: world22,joint6 exceeds the whole-attempt reaction-impulse cap for42samples(10.5ms),peak0.018727891N·m·s versuscap0.0164. Its contact-window peak0.014866956 remains below cap. The excess lies outside0..200ms contact window; aggregate JSON cannot localize it to pre-contact versus later aftermath. Do not hide this behind contact-window zeros. HeightDRstd0.1 had no recorded exceedance.

Both pilots were TRAINED with board+nail translated together uniformly±5mm. Their saved policies were evaluated at NOMINAL height via an explicitly recorded reversible reset overlay. Therefore they show nominal task learning with height randomization, not robustness at changed heights and not a replicated DR benefit.

## Measurement corrections and limits

The legacy termination name `nail_driven` maps to the200ms contact horizon, NOT the30mm depth crossing. All512native attempts themselves include at least801contact samples(801–880); whole-attempt durations300–360ms. Initial external-review warnings that30mm shortened measurement are rejected by source and saved timing evidence.

The first-strike TRACKER nevertheless finalizes its reward receipt at depth success OR its50ms deadline. This does not terminate the episode. Canonical `delivered_impulse_ns` can thus stop before the literal50ms integral; the whole-native axial integral may additionally include recontacts. These are distinct measurements, all reported separately in FINAL-RESULTS037. Example DRstd0.3at500:canonical0.3143,physicalfirst50ms0.3801,whole-native0.6073N·s. Do not present the largest as the rewarded impulse or proof that impulse is maximized. Existing normalized coefficient4 and reward-manager dt scaling were retained, so the impulse term's manager contribution is4×.02×I/Iref, not simply4×I/Iref.

The event critic observes split-strike credit state but not all peak-depth/earned-depth reward memory after rebound. Its adequacy for a fully observed value process is a hypothesis, not established by these good outcomes. Learned impedance, clipping and reward-scale mechanisms need targeted traces; this analysis does not infer them from success counts.

Zero observed violations is not safety certification, sim-to-real validation, long-horizon safety or Kyber transfer. All selected bank contexts were reused for comparison, not a held-out generalization claim. Both029combined operational closures remainFAILED/no closure.json; their independently saved evidence remains usable. All037training andscientific evaljobs completedexit0:0; failed startup checks remain preserved separately.

## Discussion recommendation

Keep the all-three-rewards-ON split/event recipe as the working baseline. Carry both std settings as valid candidates; avoid claiming a winner from near-identical aggregate outcomes. The main next discussion should address held-out height evaluation of fixed-height and DR-trained policies, margin/retained-depth/time-to-depth diagnostics, and the single load event. State-dependent std/gSDE remain separate research questions and need outcomes beyond nominal depth now that it is at ceiling. No new jobs are authorized by this analysis; no large ablation study is required to pursue the user's general constrained-contact-learning goal. Kyber grasping would adapt physical task rewards rather than copying nail impulse maximization.

## Statistical fallacy scan (11/11 checked)

Simpson/ecological: seed vectors retained, no aggregate-to-policy inference. Berkson/collider: no contact-only success selection(all512contactsknown); no adjustment on learned outcomes. Base-rate: all denominators32andUNKNOWN retained. Regression-to-mean: previous goodsplitfamilies selected, so broader selection uncertainty remains. Survivorship: all8scientificpolicies/all16cases reported, startupfailurespreserved. Look-elsewhere/forking paths: exploratory comparisons and checkpoint reuse disclosed; no p-values/significance or populationconfidence claims. Causation: bundle and physical/source equivalence limits stated, no component attribution. Reverse causality: treatment precedes learning but mechanisms remain hypotheses. StatusANALYZED, not a physics-reproduction verdict.

## Named-model discussion and adjudication

Actual providers returned completed reviews: Claude Fable5.1(`claude-fable-5-1`),Claude Opus5.5(`claude-opus-5-5`),Kimi K3.1 Pro through configuredOpenCode `opencode-go/kimi-k3.1-pro`,andGPT-6Astra Ultra(nativeindependentreview). Fable/Opus/Kimi initially received the same compactpacket withtoolsdisabled,thenallthree received the independently checked endpoint/initialization/impulse corrections. Raw and correctedresponses arepreserved underpanel/. Ultra independentlyinspectedrawJSON/source anddocumenteditsownimpulsefieldcorrection. Root compared theindependentcategorical tables:128/128 countsagree andall16canonicalmeansagreeexactly.

Aftercorrections,allreviewerssupportthecombinedall-onrecipeasaworkingbaseline,nooverallstdwinner,andnoheightrobustnessorKybertransferclaim. Fable/Opusfavorexisting-data margindiagnostics andheld-outheightprotocolnext;Kimiwithdrawsfreshconfirmatorytrainingasnecessaryafterseeingmatchedinitializationproof;Ultraemphasizestheactualrewardedsnapshotandloadexception. Rootdoesnotadoptmodelwordingthatcallsthethree taskrewardsthreeimpulserewards,norclaimsphysicalimpulsemustbereportedonlyascanonical:thethreephysicaldomainsarereportedseparately. Rootrejectsallinitialearly-depth-terminationclaims;correctedreviewswithdrawthem. Causalstrengthandall-zerooutcomesarequalifiedabove;revieweragreementisnotnewexperimentalevidence.

No reward,threshold,trainingcode,job,evaluationor nativegoalwascreated/changedforthisanalysis. Allfailedtechnicalchecksand029closurefailuresremainpreserved. The trainingmonitor cannowstopbecauseexecution,backupsandthereportarecomplete. Userreminder:assembleZ1results/checkpoints/source-environments/logsandverifiedbackups,thenplanKarolinamigrationbeforeVega maintenance2026-11-03(userreported);nomigrationperformedhere.
