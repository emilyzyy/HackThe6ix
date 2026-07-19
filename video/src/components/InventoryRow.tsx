import React from "react";
import { useCurrentFrame } from "remotion";
import { COLORS, FONT } from "../theme";
import { LegoBrick } from "./LegoBrick";
import { ramp } from "../lib/anim";

// A single inventory row that assembles from the piece itself.
export const InventoryRow: React.FC<{
  studs: [number, number];
  colorHex: string;
  part: string;
  color: string;
  qty: number;
  start: number; // local frame
  width?: number;
}> = ({ studs, colorHex, part, color, qty, start, width = 760 }) => {
  const frame = useCurrentFrame();
  const t = ramp(frame, start, start + 18);
  const pieceDrop = ramp(frame, start, start + 14);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 28,
        width,
        height: 92,
        padding: "0 26px",
        borderRadius: 8,
        background: `linear-gradient(90deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015))`,
        border: "1px solid rgba(255,255,255,0.06)",
        opacity: t,
        transform: `translateX(${(1 - t) * 40}px)`,
      }}
    >
      {/* piece render (drops in) */}
      <div
        style={{
          width: 96,
          display: "flex",
          justifyContent: "center",
          transform: `translateY(${(1 - pieceDrop) * -26}px)`,
          opacity: pieceDrop,
        }}
      >
        <LegoBrick studs={studs} unit={18} color={colorHex} />
      </div>
      <div
        style={{
          flex: 1,
          fontFamily: FONT,
          fontWeight: 700,
          fontSize: 30,
          letterSpacing: "0.04em",
          color: COLORS.offWhite,
        }}
      >
        {part}
      </div>
      <div
        style={{
          width: 150,
          fontFamily: FONT,
          fontWeight: 600,
          fontSize: 22,
          letterSpacing: "0.14em",
          color: COLORS.gray,
        }}
      >
        {color}
      </div>
      <div
        style={{
          width: 70,
          textAlign: "right",
          fontFamily: FONT,
          fontWeight: 800,
          fontSize: 30,
          color: COLORS.yellow,
        }}
      >
        ×{qty}
      </div>
    </div>
  );
};
