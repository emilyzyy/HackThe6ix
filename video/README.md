# Out of the Box — Product Demo Video

A ~2:53, 1920×1080 / 30fps product trailer built with **Remotion** (React →
deterministic timeline, CSS/SVG motion graphics). Black-and-yellow LEGO-brick
identity. No voiceover — on-screen text + background music only. Two
replaceable real-demo video slots with polished animated fallbacks.

```
video/
  src/
    demoConfig.ts        # ← single source of truth (timings, copy, colours, prices, ideas, assets)
    theme.ts             # colour + type + motion system
    Root.tsx             # Remotion composition registration
    OutOfTheBoxDemo.tsx  # scene sequencing + crossfades + music
    lib/anim.ts          # shared easing / ramp helpers
    components/          # LegoBrick, ClipSlot, TableMap, InventoryRow, AgentProgress, …
    scenes/              # ProblemReveal … FinalReveal (11 scenes)
  public/
    clips/               # scan-demo.mp4, generation-demo.mp4  (drop real footage here)
    audio/               # bg-music.mp3  (optional)
  render.mjs             # resolves asset mode + invokes Remotion
  out/out-of-the-box-demo.mp4   # rendered output
```

## Commands

```bash
cd video
npm install            # first time only

npm run preview        # open Remotion Studio to scrub/iterate
npm run render         # full 1920×1080 render -> out/out-of-the-box-demo.mp4
npm run render:draft   # fast half-res draft for quick iteration
npm run typecheck      # tsc --noEmit
```

`render.mjs` also accepts the asset-mode env var (below), e.g.
`DEMO_ASSET_MODE=mock npm run render`.

## HOW TO REPLACE THE REAL DEMO CLIPS

The composition is already built around two clip slots. You do **not** need to
touch any component to add footage — just drop the files in and re-render:

1. Export your two recordings as MP4 (H.264, 1920×1080 preferred):
   - **Scan clip** → `video/public/clips/scan-demo.mp4`
     Expected content: one combined side-by-side recording —
     LEFT = person physically scanning the pile with the phone,
     RIGHT = the live capture interface.
   - **Generation clip** → `video/public/clips/generation-demo.mp4`
     Expected content: one edited montage — Part A a suggested build, Part B a
     custom-prompt build.
2. Re-render: `npm run render` (default mode is `auto`).

The clip container, crop (`object-fit: cover`), entrance/exit, framing, corner
ticks, and status/overlay text are all already implemented and will wrap the
new footage automatically. Filenames and slot behaviour are configurable in
`src/demoConfig.ts` (`ASSETS`).

## DEMO_ASSET_MODE

Controls whether real clips or animated fallbacks are used:

| Mode | Behaviour |
|------|-----------|
| `auto` (default) | Use each real clip if it exists, otherwise render the polished animated fallback for that slot. |
| `mock` | Always render the animated fallback, even if clips exist. |
| `clips` | Require **both** real clips; fail clearly if either is missing. |

```bash
DEMO_ASSET_MODE=auto  npm run render     # default
DEMO_ASSET_MODE=mock  npm run render     # animated walkthrough only
DEMO_ASSET_MODE=clips npm run render     # hard-fail if a clip is missing
```

When a clip is absent, the fallback shows an intentional animated **system
walkthrough** (small unobtrusive `SYSTEM WALKTHROUGH` marker) — never a
`PLACEHOLDER` / `MISSING ASSET` card.

## Music

Drop `public/audio/bg-music.mp3` and it plays automatically. If the file is
missing the video still renders **silently** — the render never fails on
absent music, and no visual timing depends on the audio.

## Editing quickly (near a deadline)

Everything you'd want to change lives in `src/demoConfig.ts`:

- `SCENE_SECONDS` — per-scene durations (total is derived; keep it 2:50–2:55).
- `COPY` — every visible line of text.
- `setCards` — opening price-card names / prices / images (image `null` → an
  abstract set-box silhouette is drawn).
- `inventoryRows` — the example inventory.
- `generationIdeas` — the five suggested builds.
- `ASSETS` — clip + music paths.
- `INTENSITY` — animation volume dials.

Colours/type/motion feel live in `src/theme.ts`. Scene frame ranges are
computed as `SCENES.<name>` — no magic frame numbers in components.
