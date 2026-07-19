// ============================================================================
// Out of the Box — single source of truth for the demo video.
// Change copy, prices, durations, asset filenames, and ideas HERE.
// Nothing creative should be hunted for inside components.
// ============================================================================

import { FPS } from "./theme";

const s = (seconds: number) => Math.round(seconds * FPS);

// ---------------------------------------------------------------------------
// SCENE DURATIONS (seconds). Total is derived; keep the video 2:50–2:55.
// ---------------------------------------------------------------------------
export const SCENE_SECONDS = {
  problemReveal: 20,
  scanDemo: 21,
  understandScan: 16,
  findPieces: 8,
  fusePieces: 7,
  identifyPieces: 12,
  inventoryReveal: 14,
  questionTransition: 10,
  generationIntro: 16,
  generationDemo: 28,
  finalReveal: 21,
} as const;

export type SceneName = keyof typeof SCENE_SECONDS;

const SCENE_ORDER: SceneName[] = [
  "problemReveal",
  "scanDemo",
  "understandScan",
  "findPieces",
  "fusePieces",
  "identifyPieces",
  "inventoryReveal",
  "questionTransition",
  "generationIntro",
  "generationDemo",
  "finalReveal",
];

// Derived frame ranges: SCENES.<name> = { from, duration, to }.
export const SCENES: Record<
  SceneName,
  { from: number; duration: number; to: number; index: number }
> = (() => {
  const out = {} as Record<
    SceneName,
    { from: number; duration: number; to: number; index: number }
  >;
  let cursor = 0;
  SCENE_ORDER.forEach((name, index) => {
    const duration = s(SCENE_SECONDS[name]);
    out[name] = { from: cursor, duration, to: cursor + duration, index };
    cursor += duration;
  });
  return out;
})();

export const TOTAL_FRAMES = SCENE_ORDER.reduce(
  (acc, name) => acc + s(SCENE_SECONDS[name]),
  0
);

// ---------------------------------------------------------------------------
// ASSET PATHS (relative to video/public via staticFile()).
// ---------------------------------------------------------------------------
export const ASSETS = {
  scanClip: "clips/scan-demo.mp4",
  generationClip: "clips/generation-demo.mp4",
  music: "audio/bg-music.mp3",
} as const;

// ---------------------------------------------------------------------------
// COPY — every visible line of text, grouped by scene.
// ---------------------------------------------------------------------------
export const COPY = {
  problem: {
    kicker: "THE PROBLEM WITH A BOX OF BRICKS",
    line1: "YOU ALREADY OWN",
    line2: "HUNDREDS OF DOLLARS IN LEGO.",
    line3: "IT JUST LOOKS LIKE THIS.",
    pileLabel: "ONE FORGOTTEN BIN",
  },
  scan: {
    line1: "FIRST, WE NEED TO UNDERSTAND",
    line2: "WHAT YOU ACTUALLY HAVE.",
    statusA: "MAPPING WORKSPACE",
    statusB: "CAPTURING VIEWS",
  },
  understandScan: {
    everyCameraA: "EVERY CAMERA SEES",
    everyCameraB: "THE PILE DIFFERENTLY.",
    sameMapA: "SO WE GIVE EVERY VIEW",
    sameMapB: "THE SAME PHYSICAL MAP.",
    hundreds: "HUNDREDS OF FRAMES.",
    five: "FIVE VIEWS THAT MATTER.",
  },
  pieces: {
    fromPile: "FROM A PILE...",
    toPieces: "...TO INDIVIDUAL PIECES.",
    evenTouch: "EVEN WHEN THEY TOUCH.",
    multiA: "MULTIPLE VIEWS.",
    multiB: "ONE PHYSICAL PIECE.",
    guess: "BRICK 2×4 ?",
    verified: "BRICK 2×4",
    color: "RED",
    verifyA: "ONE GUESS",
    verifyB: "ISN'T ENOUGH.",
    evidence: ["8 STUDS", "16 × 32 MM", "3 VIEWS AGREE"],
    parallel: "10× PARALLEL RECOGNITION",
  },
  inventory: {
    fromA: "FROM A RANDOM PILE...",
    toA: "TO EVERYTHING",
    toB: "YOU ACTUALLY OWN.",
    ready: "INVENTORY READY",
  },
  question: {
    know: "NOW WE KNOW WHAT YOU HAVE.",
    become: "WHAT SHOULD IT BECOME?",
  },
  generationIntro: {
    startA: "START WITH AN IDEA",
    startB: "BUILT AROUND YOUR INVENTORY.",
    orYours: "OR START WITH YOURS.",
    promptPlaceholder: "Or ask for something completely your own…",
  },
  generationDemo: {
    pipeline: "AN AGENTIC BUILD PIPELINE",
    stages: ["PLAN", "BUILD", "REPAIR"],
    customPrompt: "a small yellow lighthouse",
  },
  final: {
    titleA: "OUT OF",
    titleB: "THE BOX",
    subA: "YOUR NEXT SET",
    subB: "IS ALREADY IN THE BOX.",
  },
  systemWalkthrough: "SYSTEM WALKTHROUGH",
} as const;

// ---------------------------------------------------------------------------
// OPENING PRICE CARDS — replace with real sets/prices/images later.
// imageSrc null => an abstract LEGO-set-box silhouette is drawn.
// ---------------------------------------------------------------------------
export const setCards: {
  name: string;
  price: string;
  imageSrc: string | null;
  color: string;
}[] = [
  { name: "SET 01", price: "$89.99", imageSrc: null, color: "#C91A09" },
  { name: "SET 02", price: "$129.99", imageSrc: null, color: "#0055BF" },
  { name: "SET 03", price: "$199.99", imageSrc: null, color: "#237841" },
];

// ---------------------------------------------------------------------------
// INVENTORY EXAMPLE ROWS — assembled from real pieces on screen.
// ---------------------------------------------------------------------------
export const inventoryRows: {
  part: string;
  color: string;
  colorHex: string;
  qty: number;
  studs: [number, number];
}[] = [
  { part: "BRICK 2×4", color: "RED", colorHex: "#C91A09", qty: 3, studs: [2, 4] },
  { part: "PLATE 1×6", color: "BLUE", colorHex: "#0055BF", qty: 2, studs: [1, 6] },
  { part: "BRICK 2×2", color: "YELLOW", colorHex: "#F2CD37", qty: 4, studs: [2, 2] },
  { part: "PLATE 2×3", color: "WHITE", colorHex: "#E8E6DE", qty: 2, studs: [2, 3] },
  { part: "SLOPE 2×2", color: "GREEN", colorHex: "#237841", qty: 1, studs: [2, 2] },
];

// ---------------------------------------------------------------------------
// SUGGESTED GENERATION IDEAS — emerge from the same inventory.
// ---------------------------------------------------------------------------
export const generationIdeas: { name: string; accent: string }[] = [
  { name: "ROVER", accent: "#C91A09" },
  { name: "DRAGON", accent: "#237841" },
  { name: "RACE CAR", accent: "#0055BF" },
  { name: "ROBOT", accent: "#F2CD37" },
  { name: "CASTLE", accent: "#E8E6DE" },
];

// ---------------------------------------------------------------------------
// ANIMATION INTENSITY CONSTANTS (dials used across scenes).
// ---------------------------------------------------------------------------
export const INTENSITY = {
  fallingBrickCount: 10, // pieces in ambient falling motifs
  pileBrickCount: 22, // bricks that make up the opening pile
  cameraFrameCount: 5, // selected viewpoints
  impliedFrameCount: 34, // representative frames used to imply "hundreds"
  parallelLanes: 10,
} as const;

// ---------------------------------------------------------------------------
// ASSET MODE — resolved by render.mjs / Root and threaded as input props.
// ---------------------------------------------------------------------------
export type AssetMode = "auto" | "mock" | "clips";

export interface DemoProps {
  assetMode: AssetMode;
  hasScanClip: boolean;
  hasGenerationClip: boolean;
  hasMusic: boolean;
}

export const DEFAULT_PROPS: DemoProps = {
  assetMode: "mock",
  hasScanClip: false,
  hasGenerationClip: false,
  hasMusic: false,
};

// Whether a given clip should render as real footage vs. animated fallback.
export const useRealClip = (
  props: DemoProps,
  which: "scan" | "generation"
): boolean => {
  if (props.assetMode === "mock") return false;
  const has = which === "scan" ? props.hasScanClip : props.hasGenerationClip;
  return has; // "auto" and "clips" both use the clip when present
};
