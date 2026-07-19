import React from "react";
import { LegoBrick } from "./LegoBrick";
import { rand } from "../lib/anim";
import { COLORS } from "../theme";

const PIECE = [
  COLORS.brickRed,
  COLORS.brickBlue,
  COLORS.brickYellow,
  COLORS.brickWhite,
  COLORS.brickGreen,
  COLORS.brickGray,
];

// A scattered pile of bricks (the "forgotten bin"). `spread` 0..1 scatters
// them; `settle` 0..1 lets them fall/settle into place.
export const BrickPile: React.FC<{
  count?: number;
  width?: number;
  height?: number;
  seed?: number;
  settle?: number; // 0 = above, 1 = settled
  unit?: number;
  style?: React.CSSProperties;
}> = ({
  count = 22,
  width = 760,
  height = 420,
  seed = 3,
  settle = 1,
  unit = 34,
  style,
}) => {
  return (
    <div style={{ position: "relative", width, height, ...style }}>
      {Array.from({ length: count }).map((_, i) => {
        const s = seed + i;
        const cx = rand(s) * (width - 120) + 20;
        const restY =
          height - 120 - Math.pow(rand(s + 2), 1.6) * (height - 160);
        const y = restY - (1 - settle) * (500 + rand(s + 3) * 300);
        const cols = 1 + Math.floor(rand(s) * 3);
        const rows = 1 + Math.floor(rand(s + 5) * 2);
        const rot = (rand(s + 1) - 0.5) * 70;
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: cx,
              top: y,
              transform: `rotate(${rot}deg) scale(${0.7 + rand(s) * 0.7})`,
              zIndex: Math.round(restY),
              opacity: settle < 0.05 ? settle * 20 : 1,
            }}
          >
            <LegoBrick studs={[cols, rows]} unit={unit} color={PIECE[i % PIECE.length]} />
          </div>
        );
      })}
    </div>
  );
};
