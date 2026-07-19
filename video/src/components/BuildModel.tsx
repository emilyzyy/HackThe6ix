import React from "react";
import { COLORS } from "../theme";
import { LegoBrick } from "./LegoBrick";
import { rand } from "../lib/anim";

// A small isometric "creation" assembled from bricks. `assemble` 0..1 flies
// pieces in and snaps them; `explode` 0..1 flies them outward. Reused for the
// generation result and the finale.
interface Piece {
  x: number;
  y: number;
  studs: [number, number];
  color: string;
}

// A compact rover-ish silhouette (art-directed, not literal).
const MODEL: Piece[] = [
  { x: 0, y: 2, studs: [4, 1], color: COLORS.brickYellow },
  { x: 0, y: 1, studs: [2, 1], color: COLORS.brickRed },
  { x: 2, y: 1, studs: [2, 1], color: COLORS.brickWhite },
  { x: 1, y: 0, studs: [2, 1], color: COLORS.brickBlue },
  { x: -1, y: 1, studs: [1, 1], color: COLORS.brickGreen },
  { x: 4, y: 1, studs: [1, 1], color: COLORS.brickRed },
  { x: 0, y: 3, studs: [1, 1], color: COLORS.brickGray },
  { x: 3, y: 3, studs: [1, 1], color: COLORS.brickGray },
];

export const BuildModel: React.FC<{
  unit?: number;
  assemble?: number; // 0..1
  explode?: number; // 0..1
  style?: React.CSSProperties;
  glow?: number;
}> = ({ unit = 46, assemble = 1, explode = 0, style, glow = 0 }) => {
  return (
    <div
      style={{
        position: "relative",
        width: 6 * unit,
        height: 5 * unit,
        ...style,
      }}
    >
      {MODEL.map((p, i) => {
        const seed = i + 1;
        const dir = rand(seed) * Math.PI * 2;
        const dist = (200 + rand(seed + 2) * 260) * (1 - assemble);
        const edist = (250 + rand(seed + 5) * 400) * explode;
        const ox = Math.cos(dir) * (dist + edist);
        const oy = Math.sin(dir) * (dist + edist) - (1 - assemble) * 120;
        const rot = (rand(seed) - 0.5) * 120 * (1 - assemble + explode);
        const op = explode > 0 ? 1 - explode * 0.15 : Math.min(1, assemble * 1.4);
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: p.x * unit + unit,
              top: p.y * unit,
              transform: `translate(${ox}px, ${oy}px) rotate(${rot}deg)`,
              opacity: op,
            }}
          >
            <LegoBrick studs={p.studs} unit={unit} color={p.color} glow={glow} />
          </div>
        );
      })}
    </div>
  );
};
