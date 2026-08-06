<!-- wayfinder-meta
{
  "schema": 1,
  "id": "wf-z1-exp-iref-propagation",
  "kind": "ticket",
  "title": "Choose how calibrated I_ref propagates into training",
  "status": "open",
  "labels": ["wayfinder:grilling"],
  "parent": {"id": "wf-z1-executable-experiments", "title": "Z1 executable experiment specification", "href": "../map.md"},
  "assignee": null,
  "claimed_at": null,
  "blocked_by": [{"id": "wf-z1-exp-drop-setup", "title": "Freeze the controlled-drop reference setup", "href": "./freeze-controlled-drop-reference-setup.md"}],
  "rank": 30,
  "created_at": "2026-08-06T22:32:04+02:00",
  "closed_at": null,
  "resolution_comment": null
}
-->

# Choose how calibrated I_ref propagates into training

## Question

When the controlled drop changes the delivered nail-impulse normalizer, should
the D4 reward coefficient remain numerically fixed, be rescaled to preserve its
effective dose, or trigger a newly matched fixed-gain baseline before downstream
comparisons?
