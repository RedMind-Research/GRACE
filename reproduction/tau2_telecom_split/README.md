# Rebuild the Tau² Telecom task split

This folder reproduces the paper's 66 held-out tasks and ten experience batches of 42 tasks each.

- **Input** is `data/tau2/domains/telecom/tasks.json` from the Tau²-bench revision pinned in [requirements/reproduction.txt](../../requirements/reproduction.txt).
- **Reference** is the published [splits.json](../tau2_telecom/data/splits.json) in this GRACE checkout.
- **Output** is `rebuilt-splits.json` in the directory passed to `--output-dir`. The example uses the Git-ignored `tau2_telecom_runs/split_rebuild/` directory.

## Get the input file

### Reuse an installed Tau²-bench checkout

If you followed the [GRACE reproduction setup](../tau2_telecom/README.md#checkout-and-environment), use the same Python environment to locate its Tau²-bench checkout.

```bash
python -m pip show tau2
```

Use the path shown as `Editable project location` and append `/data/tau2/domains/telecom/tasks.json`. Pass that full file path to `--tasks-json` in the command below.

### Download the pinned Tau²-bench checkout

If you still need Tau²-bench, run these commands from the GRACE repository root. The checkout goes into the Git-ignored `.tau2-bench/` directory.

```bash
git clone --filter=blob:none --no-checkout https://github.com/sierra-research/tau2-bench.git .tau2-bench
git -C .tau2-bench checkout --detach c5b2d228d850c59b749b93cf32c4745d3aa53967
```

## Run

Run the builder from the GRACE repository root with the core dependencies installed. If needed, install them with `python -m pip install -e .`.

The example uses the downloaded checkout. To reuse an installed checkout, set `--tasks-json` to the file path found with `pip show tau2`.

```bash
python -m reproduction.tau2_telecom_split.build \
  --tasks-json ".tau2-bench/data/tau2/domains/telecom/tasks.json" \
  --output-dir tau2_telecom_runs/split_rebuild
```

## How the split is built

1. Verify the input file and read its 2,285 task IDs.
2. Group the IDs into nine pools by intent and persona, then sort each pool by task ID.
3. Use the per-persona quotas in [config.py](config.py). Visit intents in MMS, Mobile Data, Service order and personas in None, Easy, Hard order.
4. Initialize one random generator with seed 42. For each pool, use `rng.sample()` and remove the selected IDs.
5. Draw the 66 evaluation tasks first. With the same generator, draw batches B0 through B9 from the remaining IDs in phase order `AABBAABBAA`.
6. Check each task ID, its set or batch, and its position against the published split. Write the rebuilt JSON after the check passes.
