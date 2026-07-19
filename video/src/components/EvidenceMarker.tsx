import React from "react";
import { useCurrentFrame } from "remotion";
import { COLORS, FONT } from "../theme";
import { ramp } from "../lib/anim";

// Spatial verification evidence drawn around a piece: measuring lines, stud
// markers, and short labels that converge inward. Not dashboard cards.
export const EvidenceMarker: React.FC<{
  label: string;
  angleDeg: number; // placement around the piece
  radius: number;
  start: number; // local frame
}> = ({ label, angleDeg, radius, start }) => {
  const frame = useCurrentFrame();
  const t = ramp(frame, start, start + 14);
  const converge = ramp(frame, start + 30, start + 55);
  const a = (angleDeg * Math.PI) / 180;
  const r = radius * (1 - converge * 0.14);
  const x = Math.cos(a) * r;
  const y = Math.sin(a) * r;
  return (
    <div
      style={{
        position: "absolute",
        left: "50%",
        top: "50%",
        transform: `translate(-50%,-50%)`,
      }}
    >
      {/* connector line to centre */}
      <svg
        width={Math.abs(x) * 2 + 40}
        height={Math.abs(y) * 2 + 40}
        style={{
          position: "absolute",
          left: -(Math.abs(x) + 20),
          top: -(Math.abs(y) + 20),
          overflow: "visible",
          opacity: t,
        }}
      >
        <line
          x1={Math.abs(x) + 20}
          y1={Math.abs(y) + 20}
          x2={Math.abs(x) + 20 + x * 0.82}
          y2={Math.abs(y) + 20 + y * 0.82}
          stroke={COLORS.yellow}
          strokeWidth={1.2}
          strokeDasharray="4 4"
          opacity={0.7}
        />
      </svg>
      {/* label chip */}
      <div
        style={{
          position: "absolute",
          left: x,
          top: y,
          transform: "translate(-50%,-50%)",
          opacity: t,
          whiteSpace: "nowrap",
          fontFamily: FONT,
          fontWeight: 700,
          fontSize: 20,
          letterSpacing: "0.12em",
          color: COLORS.offWhite,
          padding: "6px 12px",
          background: "rgba(0,0,0,0.55)",
          border: `1px solid rgba(255,213,0,0.4)`,
          borderRadius: 4,
        }}
      >
        {label}
      </div>
    </div>
  );
};
