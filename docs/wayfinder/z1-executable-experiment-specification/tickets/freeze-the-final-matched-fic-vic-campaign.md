<!-- wayfinder-meta
{
  "schema": 1,
  "id": "wf-z1-exp-final-fic-vic",
  "kind": "ticket",
  "title": "Freeze the final matched FIC-VIC campaign",
  "status": "open",
  "labels": ["wayfinder:grilling"],
  "parent": {"id": "wf-z1-executable-experiments", "title": "Z1 executable experiment specification", "href": "../map.md"},
  "assignee": null,
  "claimed_at": null,
  "blocked_by": [{"id": "wf-z1-exp-design-fic-rtt", "title": "Design the matched FIC RTT ablation", "href": "./design-the-matched-fic-rtt-ablation.md"}, {"id": "wf-z1-exp-estimands", "title": "Set controller-comparison estimands and margins", "href": "./set-controller-comparison-estimands-and-margins.md"}, {"id": "wf-z1-exp-iref-propagation", "title": "Choose how calibrated I_ref propagates into training", "href": "./choose-how-calibrated-iref-propagates-into-training.md"}, {"id": "wf-z1-exp-active-cat", "title": "Freeze the active impulse-CaT diagnostic", "href": "./freeze-the-active-impulse-cat-diagnostic.md"}, {"id": "wf-z1-exp-dr-matrix", "title": "Choose the curriculum and domain-randomization matrix", "href": "./choose-the-curriculum-and-domain-randomization-matrix.md"}, {"id": "wf-z1-exp-vic-mapping", "title": "Select the bounded VIC gain mapping", "href": "./select-the-bounded-vic-gain-mapping.md"}],
  "rank": 90,
  "created_at": "2026-08-06T22:32:04+02:00",
  "closed_at": null,
  "resolution_comment": null
}
-->

# Freeze the final matched FIC-VIC campaign

## Question

What exact FIC-TT versus VIC-TT treatments, shared `r_tt`, guidance, reward dose,
active CaT, curriculum/DR, seeds, evaluation populations, promotion rules, and
checkpoint-level inference contract constitute the final causal controller
comparison?
