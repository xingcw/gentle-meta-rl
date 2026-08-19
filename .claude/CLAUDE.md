# Rules

Standing instructions. These override default behavior.

## Interaction
- Always start your responses/answers with calling me "Mr. Xing".
- Do not create readme files after finishing the task. Simply show me the implementation and the readme information at the end of the task completion.
- When launching costly (>5 mins) tasks, run them in a new tmux session instead of in the background. Give me the session name so that I can potentially
check the status faster.

## Code style
- Do not write example usages into the function description.
- Do not use greek letters or non-ascii characters in the codes or comments.
- Do not add long comments when fixing codes, and try to use short-enough comments.
- Do not add yourself as a coauthor at the end of the git message when git commit.

## Config access discipline
- Never use `getattr(cfg, key, default)`, `cfg.get(key, default)`, or similar safe-loading patterns with default values when reading from OmegaConf training configs. Access config fields directly (e.g. `cfg.train.num_steps`) so that missing fields raise immediately. Silent defaults hide config bugs that are hard to detect in research experiments. This does not apply to plain Python dicts from external sources (e.g. JSON manifests).

## Changing code
- When modifying the features in @autoencoder, check if it will affect the following phase 2 training.
- Always identify the source of truth before implementing anything — the existing pattern, adapter, trainer, driver, or config that already handles the case. Search the repo for prior callers, related configs, and adjacent drivers first. If a parallel implementation already exists (e.g. `data_gen/collect_racing_axis.py` for racing-PPO collects, `data_gen/collect_ant_axis.py` for per-seed sharding), reuse and extend it; do NOT fork a new module that re-derives the same logic. Do not invent new modules, scripts, or abstractions without explicit permission. The codebase is growing faster than it should; every new file must justify why an existing one could not be extended, and that justification must be confirmed with me before the new file is written.

## Things that need my explicit permission
- Do not run `git commit` (or `git push`) until I explicitly ask. After finishing a task, leave the changes staged-or-unstaged in the working tree and stop; wait for me to say "commit" before invoking git. This applies to EVERY commit, not just the first one — an earlier "commit" does not authorize later ones. Words like "move on", "next", "continue", "do X" are NOT commit authorization; only an explicit "commit" (or unambiguous equivalent) is. If unsure whether I've authorized a commit, don't — ask.
- Do not upload/write/delete/change any files in Google Cloud Storage Bucket without explicit permissions.
