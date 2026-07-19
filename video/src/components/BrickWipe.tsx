import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { LegoBrick } from "./LegoBrick";
import { ramp, rand } from "../lib/anim";
import { COLORS } from "../theme";

const PIECE = [
  COLORS.brickRed,
  COLORS.brickBlue,
  COLORS.brickYellow,
  COLORS.brickGreen,
  COLORS.brickWhite,
];

// Signature transition: a wave of bricks rushes across the frame and obscures
// it. `progress` 0..1 (cover). Use across scene boundaries.
export const BrickWipe: React.FC<{
  start: number;
  dur?: number;
  direction?: "left" | "right";
}> = ({ start, dur = 24, direction = "right" }) => {
  const frame = useCurrentFrame();
  const p = ramp(frame, start, start + dur);
  if (p <= 0 || p >= 1) return null;
  const rows = 6;
  const dir = direction === "right" ? 1 : -1;
  return (
    <AbsoluteFill style={{ pointerEvents: "none", overflow: "hidden" }}>
      {Array.from({ length: rows }).map((_, r) => {
        const rowOffset = (rand(r) - 0.5) * 200;
        const y = (r / rows) * 1080;
        return Array.from({ length: 10 }).map((__, c) => {
          const seed = r * 11 + c;
          const lead = p * 2600 - c * 150 - rand(seed) * 120;
          const x =
            dir === 1
              ? -400 + lead + rowOffset
              : 1920 + 400 - lead + rowOffset;
          return (
            <div
              key={`${r}-${c}`}
              style={{
                position: "absolute",
                left: x,
                top: y + (rand(seed + 3) - 0.5) * 60,
                transform: `rotate(${(rand(seed) - 0.5) * 30}deg) scale(${
                  1 + rand(seed) * 0.8
                })`,
              }}
            >
              <LegoBrick
                studs={[2, 2]}
                unit={44}
                color={PIECE[seed % PIECE.length]}
              />
            </div>
          );
        });
      })}
    </AbsoluteFill>
  );
};
