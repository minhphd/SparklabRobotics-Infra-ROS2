# CLAUDE.md

Conventions for working in this repo. Read before writing code or docs here.

## Simplicity First

Minimum code that solves the problem. Nothing speculative.

    No features beyond what was asked.

    No abstractions for single-use code.

    No "flexibility" or "configurability" that wasn't requested.

    No error handling for impossible scenarios.

    If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

    Don't "improve" adjacent code, comments, or formatting.

    Don't refactor things that aren't broken.

    Match existing style, even if you'd do it differently.

    If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

    Remove imports/variables/functions that YOUR changes made unused.

    Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request. 

## Who reads this code

Researchers with a PhD-level grasp of robotics and strong Python. Write for
them. Do not explain what the code already says, do not restate a line in
English above it, and do not define standard terms (Jacobian, quaternion,
damped least squares, GIL). Python is close enough to English that most code
needs no narration at all.

## Comments

**Never more than two lines.** No exceptions, including after a section
divider — a `# ---- name ----` banner plus two lines of prose is three
consecutive comment lines and is too long. Shorten the prose, not the divider.

Comments earn their place by saying something the code cannot:

- a non-obvious constraint (`# Must match YamUltraFollowerConfig.left_server_port.`)
- a measured value's origin (`# fx = 455.570 on the D435F's native 640x360 stream`)
- a deliberate choice that looks like a mistake (`# Unguarded on purpose: ...`)
- an ordering or threading requirement that is invisible locally

Delete anything that is:

- a paraphrase of the next line
- a tutorial on a language or library feature
- a changelog (`# was 18.0, changed to ...`) — that is what git is for
- an argument for a design decision — that goes in `DESIGN.md`

Section dividers (`# ---- parking ----`) are fine and do not count as prose.

## Docstrings

A brief paragraph, then inputs, then outputs. Nothing else.

```python
def solve(self, target_pos, target_quat_wxyz, qpos_seed):
    """One step of decoupled IK. Always returns a valid 6-vector.

    target_pos: tool0 position in arm base (3,).
    target_quat_wxyz: tool0 orientation (4,).
    qpos_seed: warm start, at least 6 long.
    Returns: clamped joint positions (6,).
    """
```

The paragraph says what the function does and any single fact a caller must
know to use it correctly. It is not the place for a rationale essay, a failure
history, or a measurement table. If a parameter is self-evident from its name
and type, omit it rather than padding the list.

Module docstrings follow the same shape: what the module is, a usage snippet if
there is a CLI, and nothing else. Class docstrings document constructor
arguments in the same input-list form.

## READMEs

**A README describes a directory.** What is in it, what each file does, how to
run it, and the schemas, conventions and flags a caller needs. Tables, endpoint
lists, message formats and coordinate conventions are all description — keep
them.

**A README never explains why a design decision was made.** No "why this
exists" sections, no war stories, no "two things that will cost you an
afternoon". If that content is worth keeping it goes in `DESIGN.md` and the
README links to the relevant heading; otherwise delete it.

Keep the operational fact, move the reasoning:

> ✗ Park is all zeros, not the pose at connect — a captured pose is merely
>   wherever the arm was, so a run ending mid-air would park to mid-air and
>   then cut torque.
>
> ✓ Park is all zeros — the folded rest pose — not wherever the arm was at
>   connect. See [`DESIGN.md`](DESIGN.md#the-isaac-twin).

## DESIGN.md

The single home for rationale: why the packages are split where they are, what
each measured constant came from, which failures produced a given guard, and
what was tried and reverted. It exists so decisions are not re-litigated or
quietly undone. Nothing in it is needed to run the stack.

When you remove a rationale from a comment or README, check it is in
`DESIGN.md` first. Add a heading there if it is not.

## Changing code

Comment and docstring passes must not change behaviour. Verify by comparing the
AST before and after, ignoring docstrings — a diff that touches only prose
should prove it, not assert it.

Keep the existing style of the file you are editing: its naming, its comment
density (after the standards above), its idiom.

