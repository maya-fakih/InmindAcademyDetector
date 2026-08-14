# Roboflow pretraining pipeline (yolo26-roboflow branch)

## Goal
Pretrain yolo26n on a large, general logistics dataset, transfer the
weights for classes that match LOCO, then fine-tune on the real LOCO
subsets 2/3/5. LOCO alone is too small/noisy to get a strong backbone from
scratch in a reasonable epoch budget -- this gives the model a head start.

## LOCO's actual 5 classes
`forklift`, `pallet`, `pallet_truck`, `small_load_carrier`, `stillage`
(source: Mayershofer et al. 2020, LOCO paper)

## Datasets to merge on Roboflow
1. **Logistics** (large-benchmark-datasets/logistics-sz9jr) -- 99,238 images,
   20 classes, CC BY 4.0. https://universe.roboflow.com/large-benchmark-datasets/logistics-sz9jr
   Relevant classes to relabel:
   - `forklift` -> `forklift` (already matches)
   - `wood pallet` -> `pallet`
   - `cardboard box` -> `small_load_carrier`
2. **pallet truck** (ngc-telbo/pallet-truck-vzztc) -- 946 images, has a
   `pallet_truck` class directly. https://universe.roboflow.com/ngc-telbo/pallet-truck-vzztc
3. Need a source for `stillage` still -- a 725-image project with all 5 LOCO
   class names verbatim turned up in search but the direct URL wasn't
   captured; re-search Roboflow Universe filtered by `class:stillage` before
   relying on this.

## Explicitly avoid
Do not use any Roboflow project with "LOCO" in the name/description as a
pretraining source -- high risk it's a re-upload or subset of the actual
TUM LOCO images, which could silently include subset-1/4 holdout images.

## Steps
1. In Roboflow, fork/merge datasets 1 and 2 above (plus a stillage source
   once found) into one workspace.
2. Relabel classes per the mapping above so class names match LOCO exactly.
3. Export as YOLO format.
4. **Before training on it**: run the leak check --
   ```
   uv run scripts/dedup_check.py \
       --holdout dataset/subset-1 dataset/subset-4 \
       --candidates path/to/roboflow_export/images \
       --threshold 5
   ```
   Review `flagged.txt` by eye -- delete confirmed duplicates from the
   export before merging further. This only catches near-identical images,
   not different photos of the same scene; residual leak risk isn't fully
   eliminated and should be stated as a limitation, not claimed as solved.
5. Train yolo26n on the merged, deduped, relabeled dataset (separate from
   the LOCO training config -- this is a pretraining run, not the final
   fine-tune).
6. Once you have that checkpoint, verify `models/class_weight_transfer.py`
   against it before using it for real:
   ```python
   from models.class_weight_transfer import list_candidate_tensors
   print(list_candidate_tensors(pretrained_model, num_classes=5))
   ```
   Confirm these are actually the classification output layers (not
   box-regression) before calling `transfer_matched_class_rows`.
7. Fine-tune the transferred checkpoint on LOCO subsets 2/3/5 as usual.

## Known gap
No pretrained-weight transfer available for `pallet_truck` or `stillage`
unless step 3 above turns up a usable source -- those two heads stay
randomly initialized either way, same as the current baseline behavior.
