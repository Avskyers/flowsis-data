# FlowSIS Data

## Installation

You will need Python and Git to use this repo. If you do not have Python, you can try and install it. It is easiest on Linux and MacOS. On Windows, I recommend using WSL2 to emulate Linux. If you are comfortable with a terminal and command line I recommend installing Python with Miniforge, otherwise you can use Anaconda, which has a graphics user interface (GUI). Installation for both of these is relatively simple. If you cannot install Python use the lab PC, which runs Linux. Similarly, install Git based on your system.

Once you have Python and Git, in your command line (called Terminal in MacOS and Linux) type:

```shell
# Navigate to home. If you prefer to clone to somewhere else, nagivate there.
cd ~
# Clone the repo to your local machine
git clone https://github.com/pricyspark/flowsis-data.git
cd flowsis-data

conda env create -f environment.yml -y
conda activate flowsis-data
```

## Usage

We will use Git pushes and pulls to sychronize data. I recommend briefly researching the basics of what Git is, and what pushes and pulls do. If you already have a preferred Git GUI, use that. Otherwise, this guide will give command line interface (CLI) instructions to enter into the terminal.

Download your original images, unzip them if needed, and place them in
`raw-images`. For single-tool captures, use the physical Tool ID as the folder
name. For multiple tools, use the reserved `multi` folder with a subdirectory
for each photographed arrangement:

```text
raw-images/
  12/                         # Single-tool captures of physical tool 12
  13/                         # Single-tool captures of physical tool 13
  multi/
    setup-001/                # One unchanged arrangement across rotations
      IMG_1001.HEIC
      IMG_1002.HEIC
    setup-002/                # A different arrangement
      IMG_1010.HEIC
```

Use unique setup folder names. Moving or flipping a tool, changing its opening,
changing the lighting, or adding/removing tools starts a new setup. Rotating the
unchanged arrangement stays in the same setup. Folder names organize captures;
physical tool identities for multi-tool images belong in the object manifest.
The converter preserves subdirectories, so
`raw-images/multi/setup-001/IMG_1001.HEIC` becomes
`images/multi/setup-001/IMG_1001.jpg`.

Pull any incoming changes before processing:

```shell
git pull
```

Convert and crop the raw images. The converter isolates the blue marker in each image quadrant, matches its inner corner with an L-shaped convolution kernel, applies a perspective correction, and saves the finished 2160×2160 JPG in `images`. If a marker is missing, incomplete, or does not match the kernel reliably, it does not create the JPG and lists the failed image and reason after processing finishes.

```shell
python img_convert.py
```

Existing JPGs are skipped by default. To reprocess all raw images and replace their existing JPGs, run:

```shell
python img_convert.py --force
```

If forced conversion fails, the old JPG is removed so `images` cannot retain an output that failed the current crop validation. The original file remains in `raw-images`.

To save annotated previews of the detected coordinates for visual review, run:

```shell
python review_crop_coordinates.py --count 32
```

Use `--count 0` to review every raw image. Previews are saved in `coordinate-review` and are not used by the converter.

Prepare the image and object manifests (this script was previously named `basic.py`):

```shell
python prepare_manifests.py
```

For single-tool folders, the script lists each new tool ID and asks you for its class, name, and confidence. Tool classes and names are automatically changed to lowercase with punctuation removed. Confidence must be a number from 0 to 1. If it finds undocumented images for a tool ID that is already in the manifest, it warns you and asks whether to add or skip them. Pressing Enter skips that tool. If you continue, the script reuses the tool's existing class, name, and confidence. It reads the original files in `raw-images` to fill in the capture time, camera make, model, flash, ISO, exposure, aperture, and focal length in millimeters when that metadata is available. Capture times are stored with their date and timezone, for example `2026-09-01T19:51:50.430-04:00`. The metadata is written only to the CSV; it is not copied into the JPG files.

`multi/` is reserved for explicit intake. The preparation script never treats
`multi` as a Tool ID, creates inferred instances there, or assigns sample IDs
and angles from its filenames. It reports converted multi-tool images that have
not been registered, and registered multi-tool images with no instance rows,
with instructions to run `annotate.py declare`. These images do not interrupt
single-tool preparation. Declared multi-tool images still receive available
camera metadata and refreshed navigation columns; their identities, masks,
sample IDs, and angles are preserved.

Use the two manifests as spreadsheets. The columns you need for manual review
are at the left; internal IDs and mask details are farther to the right.

| File | What to check and edit |
| --- | --- |
| `image_manifest.csv` | Sample ID, Angle, Filepath, Lights, and other capture details. |
| `image_object_manifest.csv` | Tool ID, Tool Class, Tool Name, and identification Confidence for each tool in an image. |

The image manifest starts with `Tool IDs`, `Sample ID`, `Angle`, `Filepath`, and
`Lights`. `Tool IDs` is a summary: `0` means one tool, and `0;12` means tools 0
and 12. Change physical tool identities in the **object manifest**, then run
`python prepare_manifests.py` to refresh this summary.

The object manifest starts with tool ID, class, name, confidence, sample ID,
and filepath. Its sample ID and filepath are copied from the image manifest
so you can find an image without looking up its internal ID. Edit these two
fields in the **image manifest**, then run `python prepare_manifests.py` to refresh the
copies. Do not edit internal Image ID or Instance ID values. Verification
reports summaries or references that disagree between files.

A **Tool ID** identifies one physical instrument. A **Sample ID** identifies
one setup photographed across rotations. Flipping a tool, changing its opening,
moving it to a new location, changing the lighting, or otherwise changing the
setup starts a **new sample**. Rotating the unchanged setup stays in the same
sample. For multiple tools, Angle describes rotation of the entire arrangement.

Keep `Angle` in the image manifest. The eight views are 0, 45, 90, 135, 180,
225, 270, and 315 degrees; 360 repeats 0. If filenames are out of order, correct
the angles beside the corresponding filepaths. For example:

| Tool IDs | Sample ID | Angle | Filepath |
| --- | --- | --- | --- |
| 0 | 15 | 0 | images/0/IMG_1002.jpg |
| 0 | 15 | 45 | images/0/IMG_1003.jpg |
| 0 | 15 | 90 | images/0/IMG_1001.jpg |

Run `python prepare_manifests.py` after editing. It puts unfinished images (missing Sample ID or Angle)
at the bottom, and sorts each section by numeric Tool IDs. Rows with the same
tool IDs retain their existing order, regardless of sample, angle, or filename.
Object rows follow the same image order. It preserves
existing sample IDs and angles. Filename order only provides initial suggestions
for new, unassigned batches; check that each batch really is one unchanged setup.

`prepare_manifests.py` creates a pending instance for each new single-tool image and reuses
existing tool details from object rows. Annotation fields stay blank until review;
identification `Confidence` is separate from `Annotation Confidence`. Existing
masks and review fields are preserved when adding images. Commit both manifests
together.

Multiple tools can be represented by additional instance rows with the same
`Image ID` and distinct `Instance ID` values. Automatic folder intake remains
for single-tool captures. For multi-tool images, use `annotate.py declare` to
add explicit tool metadata, then run `annotate.py images` without `--single-tool`
for mask-to-tool assignment. Metadata-only instance rows are excluded from mask
training. An accepted empty image retains its tool metadata without a positive
mask annotation.

For single-tool folders, raw images are considered in consecutive filename slots of eight, including newly added raw images that do not have a converted JPG yet. A complete slot receives a unique sample ID and angles from 0 to 315 degrees. If conversion or cropping fails, the raw file continues to occupy its original slot. The other images in that slot are left ungrouped for review, but the failure does not shift the angles or prevent later complete eight-image slots from being assigned automatically.

Check both `image_manifest.csv` and `image_object_manifest.csv`. Confirm instance tool identities and confidence, image filepaths, sample groupings, and angles. In particular, confirm that each recorded Angle matches its image, regardless of filename order.

You can also run an automatic verification:

```shell
python verify_data.py
```

The verifier first checks the CSV and files for missing images, images that are not in the manifest, duplicate paths, tool IDs that disagree with their folder, invalid confidence values, incomplete samples, and missing or repeated angles. It then asks whether Codex should visually review temporary, downscaled contact sheets. The visual review looks for a different tool within a sample, suspicious rotation order, image-quality problems, and inconsistent class or name conventions. These findings are suggestions for a person to review; the verifier never changes the manifest. Temporary contact sheets are deleted automatically.

The visual review requires the Codex command-line program and a one-time ChatGPT sign-in. If `codex` is not installed, install it with:

```shell
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex login
```

The results are saved as `verification_report.html`, which you can open in a web browser, and `verification_report.json`, which contains the same results in a machine-readable format. To run only the CSV and file checks without Codex, use `python verify_data.py --static-only`.

Once the manifest is correct, use Git to synchronize it with GitHub. Before pushing, pull again to check for incoming changes.

```shell
git pull
```

Add the files we want to synchronize to GitHub:

```shell
git add *
```

The files are now "staged". We commit staged changes so they are saved to the Git history. Each commit needs a message to describe the change. Format your commit message using your name, the tools, and the samples added:

```shell
git commit -m "Jane Doe" -m "tools: x, y, z" -m "samples: a, b, c, d, e, f"
```

Finally, push it to GitHub to upload it online so everyone can now see them.

```shell
git push
```

You can now check the repo on the GitHub website to confirm your changes.

## Mask annotation

After converting images and updating the manifest with `prepare_manifests.py`, generate and
review SAM3 masks from the `flowsis-data` repository root:

```shell
# Update an existing environment after pulling the annotation changes.
conda env update -n flowsis-data -f environment.yml
conda activate flowsis-data
python annotate.py images --single-tool
```

The first run downloads `facebook/sam3` through Transformers. If access to the
checkpoint requires authentication, authenticate with Hugging Face using an
account authorized to download it. Interactive review requires a graphical
Matplotlib backend. CUDA is selected when available; use `--device cpu` to
select CPU explicitly. `--batch-size 1` reduces the image batch size.

`--single-tool` merges fragments such as the two sides of open scissors into
one object. Omit it for multi-tool images. Prompts use `Tool Class` by default;
use `--image-prompt name-and-class` for prompts such as "iris scissors".
A second pass refines a padded crop by default; disable it with
`--no-refine-crop`.


In the terminal review prompt:

- `a` accepts all candidates; `a 0,2` accepts selected candidates.
- `e` accepts an empty image; `r` rejects it; `s` skips it for later.
- `p TEXT` reruns segmentation with another prompt.
- `i 0,2` marks ignore regions; `v partial 0` sets an instance's visibility
  (`clear`, `partial`, or `severe`).
- `q` quits. Rerunning the same command resumes unfinished images.

Previously skipped images are shown after other unfinished images. Each group
keeps its manifest order, and each image is shown at most once per run. To work
through skipped images first, run:

```shell
python annotate.py images --single-tool --skipped-first
```

Basic preparation assigns `Image ID` and `Capture Session ID`. Annotation reads
tool metadata from instance rows. Review the suggested capture sessions before splitting
training data. Outputs are `image_object_manifest.csv`, `masks/images/<tool ID>/<image stem>.npz`,
and the append-only `image_review.jsonl`. Commit these together with the updated
`image_manifest.csv` so masks and resume decisions stay synchronized. Mask
bundles retain the format consumed by `flowsis` training. `--auto-accept` bypasses
human review and should only be used when that is intended.

### Multiple tools and overlapping tools

Before mask annotation, prepare multi-tool captures in this order:

1. Place originals under `raw-images/multi/<setup>/`, keeping each unchanged
   arrangement in its own setup folder.
2. Run `python img_convert.py` and inspect any conversion failures.
3. Declare every physical tool in each converted image. Repeat `--filepath`
   to declare the same tool across several images of an arrangement:

   ```shell
   python annotate.py declare \
     --filepath images/multi/setup-001/IMG_1001.jpg \
     --filepath images/multi/setup-001/IMG_1002.jpg \
     --tool-id 12 --tool-class scissors --tool-name iris --confidence 0.9
   python annotate.py declare \
     --filepath images/multi/setup-001/IMG_1001.jpg \
     --filepath images/multi/setup-001/IMG_1002.jpg \
     --tool-id 13 --tool-class forceps --tool-name fine --confidence 0.9
   ```

   Include all converted images of the arrangement, not just the two shown in
   this example. `declare` adds image rows if necessary and records the actual
   tool identities without inferring them from folder names.

4. In `image_manifest.csv`, assign one globally unique Sample ID to the unchanged
   arrangement and enter each image's actual Angle. Use the eight views from 0
   to 315 degrees for a complete rotation sample. A setup folder name does not
   automatically assign a Sample ID. Review each tool's ID, class, name, and
   identification confidence in `image_object_manifest.csv`.
5. Run `python prepare_manifests.py` to fill available camera metadata and
   refresh navigation columns. It preserves your manual sample and angle values.
6. Run `python verify_data.py --static-only` and resolve manifest errors before
   annotation.
7. Run `python annotate.py images` to assign and review masks.

You may run `prepare_manifests.py` before step 3; it prepares single-tool captures
and lists the multi-tool images that still need declaration. It does not infer
how many tools are present: reviewers must declare **every** physical tool, even
if an image already has one instance row.

Instead of repeated filepaths, use `--sample-id 15` to declare a tool in every
image of an existing, manually assigned rotation sample. Instance IDs are
generated from the image and physical tool ID; `--instance-id` allows an explicit
identifier. Duplicate instance or physical tool IDs within an image are rejected.
`declare` does not infer a rotation sequence or camera metadata.
To add a tool to an already reviewed image, `declare --reopen` clears all its
annotation fields and reopens it for review, while keeping existing identities.

Run multi-tool annotation **without** `--single-tool` or `--auto-accept`.
The model runs each unique declared label once per image, so two scissors share
a prompt but still require separate identity assignments. Optional
`--discovery-prompt` and `--canonical-prompt` add predictions. All candidates are
retained for explicit review, including possible duplicates and overlapping
masks. Automatic crop refinement is disabled for batches containing multi-tool
images and for additional prompts entered during explicit review. Per-instance
refinement after assignment is deferred.

The interface uses a Matplotlib image preview and terminal commands. The preview
shows colored, numbered masks, bounding boxes, prompts, model scores, and current
assignments. The terminal lists declared Instance IDs, Tool IDs, classes, names,
identification confidence, visibility, and assigned candidate indices. Prompts
and scores are suggestions; the reviewer supplies the physical identity.

The explicit reviewer is used by default. `--single-tool` retains the original
single-tool review commands when there is one declared tool and one candidate;
ambiguous results switch to explicit assignment. Its fragment union is only
appropriate for captures known to contain one tool.

| Command | Action |
| --- | --- |
| `g INSTANCE_ID 0,2` | Assign or combine candidates 0 and 2 into this tool's mask; replaces its previous assignment. |
| `u INSTANCE_ID` | Clear an assignment and its visibility state. |
| `v INSTANCE_ID partial` | Set `clear`, `partial`, `severe`, `occluded`, or `absent`. Occluded means fully hidden; absent means not in the image. |
| `x 1,3` | Exclude false or duplicate candidates. |
| `i 1,3` | Add candidates to the shared ignore region. |
| `p forceps` | Append predictions using another prompt while preserving existing assignments. |
| `n INSTANCE_ID TOOL_ID "CLASS" "NAME" CONFIDENCE` | Declare an unexpected tool with complete metadata. Quote multiword class/name values. |
| `a` | Accept after every tool and candidate is resolved. |
| `e` | Accept an empty image: explicitly mark all declared tools absent and exclude remaining candidates. |
| `s`, `r`, `q` | Skip, reject, or quit. |

A candidate can be assigned to only one tool. Clear its assignment before moving
it to another tool or excluding it. Each candidate must be assigned, excluded,
or ignored; each tool must have a visible mask or be explicitly occluded/absent.
Use `partial` or `severe` when some of the tool remains visible. Acceptance does
not infer physical identities or create unnamed reviewed tools.

Final masks describe **visible surfaces**. Grouping disconnected fragments takes
their pixelwise union without filling gaps or inventing hidden geometry. The
result occupies one mask entry and one object row, with a bounding box around
all its visible fragments. Separate tools retain independent masks, including
overlapping pixels when supplied by the reviewer. Review the crossing regions
carefully to avoid labeling hidden surfaces.

The CSV columns and version-2 `[N, H, W]` mask-bundle layout are unchanged.
Visible rows reference their declared Instance ID and exact bundle index. Absent
and fully occluded tools retain metadata and resolved visibility but have no
positive mask, blank mask paths/boxes, and index `-1`. They use
`Annotation Complete=TRUE` and `Reviewed=FALSE`: there is no positive mask to
review, and this metadata-only convention allows the existing flowsis training
builder to consume the image's visible tools. The review log records the human
resolution of every tool, including these nonvisible tools.

Assignments and fragment grouping are recorded in `image_review.jsonl`, with
source candidate IDs, indices, prompts, scores, and per-tool visibility. The
final mask bundle does not preserve fragments separately. Skipped/quit images
restart candidate generation and assignments on the next run; the log retains
previous actions for audit but does not restore an unfinished editing session.
The verifier checks completed rows for unresolved instances, invalid flags,
mask identity/index mismatches, bounding boxes, and image dimensions.

### Single-tool video prototype

Put MP4 originals in `videos/<Tool ID>/`, for example `videos/12/opening.mp4`.
Each video must contain at most the one known physical instrument, already
identified in `image_object_manifest.csv`. The tool can be absent or fully
occluded in individual frames. Conflicting image class/name metadata is reported
for correction; identification confidence uses the lowest existing confidence
for that physical tool. Files directly in `videos/` require an explicit
`--tool-id` during preparation.

Update the environment to install PyAV, then extract every frame:

```shell
conda env update -n flowsis-data -f environment.yml
conda activate flowsis-data
python annotate.py prepare-videos
python annotate.py verify-videos
python annotate.py videos
```

Preparation does not download model weights. It writes PNG images at
`frames/videos/<Video ID>/000000.png`, `000001.png`, etc., preserving decoded
RGB pixels without resizing or cropping. Source display-rotation metadata is
not applied. Each frame's decoded presentation timestamp is stored explicitly;
frame times are not calculated from FPS. Audio is not extracted.

Two CSVs are authoritative:

| Manifest | Rows and fields |
| --- | --- |
| `video_manifest.csv` | One row per MP4: Tool ID/class/name/confidence, Video ID, source Filepath, FPS, actual decoded Frame Count, Width/Height, Capture Session ID, Condition, source SHA256, Extraction ID, and Extraction Complete. |
| `video_frame_manifest.csv` | One row per frame: Video ID, zero-based Frame Index, Timestamp Seconds, PNG Filepath, Frame ID, PNG SHA256, bounding box, Mask Path/Index, visibility, annotation/review flags, Review Status, and optional Anchor Path. Tool identity comes from the video row. |

There is no separate video object manifest in this single-tool prototype.
`Extraction Complete` must be true before annotation. Extracted frames initially
have blank annotation fields and pending review status. Capture sessions default
to distinct video session IDs; use `--capture-session-id` during preparation or
edit the video manifest to group captures from the same actual session.
Physical Tool IDs are retained for training split grouping.

Rerunning preparation verifies and reuses unchanged PNGs. Missing PNGs can be
recreated exactly from the unchanged MP4. A changed MP4 or modified PNG requires
`python annotate.py prepare-videos --reextract`; this clears affected masks,
anchors, and review decisions before extracting again. Use `--video-id VIDEO_ID`
to restrict preparation, annotation, verification, or export to one video.
Interrupted extraction resumes on the next preparation run.

The reviewer shows one frame with a green current mask, plus numbered text
candidates when requested. Start with `p` to use the known tool's name/class,
then `g 0` to select a candidate (`g 0,2` combines fragments). Alternatively,
use mouse prompts directly. A missing prediction remains unresolved; it is never
silently labeled absent or occluded. Text prompts change segmentation suggestions,
not the physical identity or training label.

| Command | Action |
| --- | --- |
| `p [TEXT]` | Generate text candidates for this frame, or the selected repair region. Omit TEXT to use the current tool prompt. |
| `g 0,2` | Select/group candidates into the current tool mask. The union preserves disconnected visible pieces and empty gaps. |
| `click` | Open the mouse editor: left click marks tool pixels; right click marks background. Clicks accumulate for the same frame and repair region during the reviewer session. Backspace undoes the last click; Enter generates a replacement; Esc cancels. Include positive examples of all desired visible parts. |
| `box` | Draw a box to guide segmentation. Enter applies it; Esc cancels. |
| `region` or `region L T R B` | Select a repair rectangle by dragging or providing pixel coordinates. Right/bottom edges are exclusive. Text, click, and box prompts operate on this crop; only pixels inside it are replaced. |
| `whole` | Return to whole-tool correction. Whole-tool prompts can change pixels anywhere in the frame. |
| `prop both 80 160` | From the current corrected frame, propagate backward and forward within the inclusive range 80–160. `forward` and `backward` are also supported. |
| `preview 80 160` | Inspect every frame's current mask in pages of up to 12 overlays. Close each page to continue. |
| `accept 80 160` | Accept nonempty masks in that inclusive range after previewing those exact mask versions. Changed masks require another preview. |
| `v partial`, then `a` | Set `clear`, `partial`, or `severe` and accept the current visible mask. `a` defaults to clear when no state was set. |
| `absent` / `occluded` | Explicitly accept this frame with no visible mask, distinguishing absence from a fully hidden tool. |
| `reopen 80 160` | Reopen reviewed frames in this inclusive range before replacing their masks. Existing corrected anchors remain available as inputs. |
| `jump 120` | Inspect another frame directly. |
| `next` / `s`, `r`, `q` | Leave this frame unfinished and continue, reject it, or quit the entire video command. |

For example, if early masks are good but tracking drifts around frame 120:

1. `jump 120`, then `reopen 80 160` if those frames were already reviewed.
2. Use `click` or region-only repair and inspect the resulting mask.
3. Run `prop both 80 160`.
4. Run `preview 80 160`, correct any remaining failures, then accept the good
   frames or ranges. Resolve missing masks explicitly.

Propagation uses the corrected **whole-tool** mask, including region-only repairs.
A repair rectangle itself does not follow the moving tool. The model can read
neighboring frames for context, but only the requested range is replaced.
Accepted/rejected frames are protected until explicitly reopened. Corrected
anchor masks are saved separately and reused together within each model window,
including after restarting the reviewer. Click markers are retained during the
current session; after a restart, the saved correction mask remains available
but the markers are not restored in the editor. Each direction is processed in bounded
chunks with overlapping context and a carried boundary mask. Defaults are
`--chunk-size 64 --context-frames 8`; lower them to reduce memory use. Tracking
can still drift across chunk boundaries or lose the tool; new predictions remain
pending human review. Add another anchor where tracking fails.

Final masks are independent version-2 bundles at
`masks/videos/<Video ID>/000000.npz`. A visible tool has one mask and identity
`tool-<Tool ID>`; absent/occluded frames have empty bundles. Each PNG and NPZ can
be loaded independently without opening the MP4. The append-only
`video_review.jsonl` records decisions, grouping, click coordinates (relative
to the crop when repairing a region), regions, extraction generation, and
propagation ranges. Final masks do not retain fragments separately.

`--auto-accept` is an explicit bypass for existing nonempty pending predictions;
it does not generate missing masks or infer absence. The first version uses
Matplotlib windows plus terminal commands, not a timeline editor. Its segmentation
and tracker backends load `facebook/sam3` on demand.

To train from extracted frames with the existing flowsis image reader, export an
image-style snapshot:

```shell
python annotate.py export-video-training
cd video_training
flowsis-build-annotated-dataset \
  --image-manifest image_manifest.csv \
  --object-manifest image_object_manifest.csv \
  --review-log image_review.jsonl \
  --image-mask-dir masks/images \
  --video-review-log /dev/null \
  --output-path ../../flowsis/data/new/video-segmentation-dataset
```

The export keeps frame IDs, physical Tool IDs, and capture sessions. It includes
accepted clear/partial frames and explicit absent frames as negatives; severe
and fully occluded frames remain in the annotation manifests but are excluded
from this base-training snapshot. It writes one PNG/NPZ per included frame,
using hard links where possible and copies across filesystems. Atomic source
replacement cannot change an existing snapshot. Re-export after corrections;
use `--replace` to refresh an existing export or choose a new `--output-dir`.
The existing builder requires enough independent tool/session groups for its
requested splits; for a one-tool prototype use `--validation-size 0 --test-size 0`.
It reads the exported frame files, without decoding or needing the original MP4.
Do not pass the new video review log to the legacy MP4 training branch.

Use `python annotate.py --help` and each command's `--help` for options.

### Build training data in flowsis

The existing training builder resolves mask paths against its working directory.
Keep running from the `flowsis-data` root, including after moving or cloning this
repository, and use the environment where `flowsis` is installed:

```shell
conda activate flowsis
flowsis-build-annotated-dataset \
  --image-manifest image_manifest.csv \
  --object-manifest image_object_manifest.csv \
  --review-log image_review.jsonl \
  --image-mask-dir masks/images \
  --video-review-log /dev/null \
  --output-path ../flowsis/data/new/segmentation-dataset
```

Paths stored by annotation are relative to the directory where it runs, so use
the repository root consistently. Custom output directories must also remain
accessible from that directory.

Run annotation tests without downloading model weights:

```shell
python -m pytest tests -q
```

Mask paths mirror the image tree: `images/0/IMG_8585.jpg` produces
`masks/images/0/IMG_8585.npz`. Image IDs and review-log keys remain unchanged.
Images with the same stem in the same folder but different extensions must
be renamed to avoid mask collisions.

## Uninstallation

```shell
# If you cloned into a directory other than home, go there instead.
cd ~

rm -rf flowsis-data
```
