# Local Markdown Wayfinder tracker

This repository has no configured external issue tracker, so Wayfinder maps use
Markdown files under `docs/wayfinder/`.

Each map and ticket starts with a `wayfinder-meta` HTML comment containing strict
JSON. Internal IDs support validation, but all user-facing references use the
issue title as linked text.

## Layout

```text
docs/wayfinder/<map-name>/map.md
docs/wayfinder/<map-name>/tickets/<ticket-name>.md
docs/wayfinder/<map-name>/comments/<ticket-name>/<timestamp>-resolution.md
```

## Metadata contract

- A map has `kind: "map"`, label `wayfinder:map`, and no parent.
- A ticket has `kind: "ticket"`, one `wayfinder:<type>` label, a named parent,
  `assignee`, `blocked_by`, and a numeric `rank`.
- A resolution comment has `kind: "comment"`, `comment_type: "resolution"`, and
  a named ticket link.
- `assignee: null` means unclaimed. Claiming a ticket is the first mutation in a
  work session.
- A ticket is closed only after its resolution comment exists, `status` is
  `closed`, `closed_at` is populated, and `resolution_comment` points to the
  named comment.

## Frontier query

The frontier is the set of tickets that are:

1. `open`;
2. unassigned;
3. blocked only by tickets whose metadata says `closed`.

Sort frontier tickets by `rank`, then title. Resolve each dependency through its
`href`; never trust an ID without checking the linked ticket. Render results as
linked titles, never bare IDs.

## Map updates

On resolution:

1. write the resolution comment;
2. close the ticket and link that comment;
3. append one linked one-line gist to the map's `Decisions so far`;
4. create newly visible tickets first and wire dependencies in a second pass;
5. remove any corresponding text that graduated from `Not yet specified`.

If a ticket proves out of scope, close it and link it only from `Out of scope`,
not from `Decisions so far`.
