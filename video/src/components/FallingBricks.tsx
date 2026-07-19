import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { LegoBrick } from "./LegoBrick";
import { rand } from "../lib/anim";
import { COLORS } from "../theme";

const PIECE_COLORS = [
  COLORS.brickRed,
  COLORS.brickBlue,
  COLORS.brickYellow,
  COLORS.brickWhite,
  COLORS.brickGreen,
  COLORS.brickGray,
];

// Ambient pieces falling under gravity. Quiet, cinematic, purposeful.
export const FallingBricks: React.FC<{
  count?: number;
  speed?: number;
  seed?: number;
  opacity?: number;
  unit?: number;
}> = ({ count = 10, speed = 1, seed = 7, opacity = 1, unit = 30 }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ opacity }}>
      {Array.from({ length: count }).map((_, i) => {
        const r = rand(i + seed);
        const r2 = rand(i * 3 + seed);
        const r3 = rand(i * 7 + seed);
        const x = r * 1920;
        const period = 240 + r2 * 180;
        const startOffset = r3 * period;
        const t = ((frame * speed + startOffset) % period) / period; // 0..1
        const y = t * 1320 - 200;
        const rot = (r2 - 0.5) * 90 + frame * (0.4 + r3) * (r > 0.5 ? 1 : -1);
        const scale = 0.5 + r2 * 0.9;
        const cols = 1 + Math.floor(r * 2);
        const rows = 1 + Math.floor(r3 * 3);
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: x,
              top: y,
              transform: `rotate(${rot}deg) scale(${scale})`,
              opacity: 0.35 + r2 * 0.5,
            }}
          >
            <LegoBrick
              studs={[cols, rows]}
              unit={unit}
              color={PIECE_COLORS[i % PIECE_COLORS.length]}
            />
          </div>
        );
      })}
    </AbsoluteFill>
  );
};
