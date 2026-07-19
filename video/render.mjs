#!/usr/bin/env node
// Out of the Box — demo video renderer.
//
// Resolves DEMO_ASSET_MODE (auto | mock | clips), detects which real clips and
// music exist, threads them as input props, and invokes the Remotion CLI.
//
//   DEMO_ASSET_MODE=auto  node render.mjs            (default)
//   DEMO_ASSET_MODE=mock  node render.mjs            (always animated fallback)
//   DEMO_ASSET_MODE=clips node render.mjs            (fail if a clip is missing)
//   node render.mjs --draft                          (fast low-res preview)

import { execSync } from "child_process";
import { existsSync, mkdirSync, writeFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const publicDir = join(__dirname, "public");
const outDir = join(__dirname, "out");

const mode = (process.env.DEMO_ASSET_MODE || "auto").toLowerCase();
if (!["auto", "mock", "clips"].includes(mode)) {
  console.error(`✗ Invalid DEMO_ASSET_MODE="${mode}" (use auto | mock | clips)`);
  process.exit(1);
}
const draft = process.argv.includes("--draft");

const scanClip = join(publicDir, "clips", "scan-demo.mp4");
const genClip = join(publicDir, "clips", "generation-demo.mp4");
const music = join(publicDir, "audio", "bg-music.mp3");

const hasScan = existsSync(scanClip);
const hasGen = existsSync(genClip);
const hasMusic = existsSync(music);

// In "clips" mode both real clips are mandatory — fail clearly.
if (mode === "clips") {
  const missing = [];
  if (!hasScan) missing.push("public/clips/scan-demo.mp4");
  if (!hasGen) missing.push("public/clips/generation-demo.mp4");
  if (missing.length) {
    console.error("✗ DEMO_ASSET_MODE=clips requires both real clips. Missing:");
    for (const m of missing) console.error(`    - ${m}`);
    console.error("  Add the clips or use DEMO_ASSET_MODE=auto / mock.");
    process.exit(1);
  }
}

// In "mock" mode we always render the animated fallback, even if clips exist.
const useScan = mode === "mock" ? false : hasScan;
const useGen = mode === "mock" ? false : hasGen;

const props = {
  assetMode: mode,
  hasScanClip: useScan,
  hasGenerationClip: useGen,
  hasMusic,
};

console.log("Out of the Box — video renderer");
console.log("─".repeat(52));
console.log(`  mode           ${mode}${draft ? "  (draft)" : ""}`);
console.log(`  scan clip      ${useScan ? "✓ real footage" : "○ animated walkthrough"}`);
console.log(`  generation     ${useGen ? "✓ real footage" : "○ animated walkthrough"}`);
console.log(`  music          ${hasMusic ? "✓ present" : "○ silent (ok)"}`);
console.log("─".repeat(52));

mkdirSync(outDir, { recursive: true });
const propsFile = join(outDir, ".render-props.json");
writeFileSync(propsFile, JSON.stringify(props, null, 2));

const outputPath = join(outDir, "out-of-the-box-demo.mp4");

const cmd = [
  "npx",
  "remotion",
  "render",
  "src/index.ts",
  "OutOfTheBoxDemo",
  outputPath,
  `--props=${propsFile}`,
  "--overwrite",
  draft ? "--scale=0.5 --jpeg-quality=70 --crf=30" : "--crf=18",
]
  .filter(Boolean)
  .join(" ");

console.log(`\nRunning: ${cmd}\n`);

try {
  execSync(cmd, { stdio: "inherit", cwd: __dirname });
  console.log(`\n✓ Rendered: ${outputPath}`);
} catch (err) {
  console.error("\n✗ Render failed:", err.message);
  process.exit(1);
}
