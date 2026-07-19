import React from "react";
import { COLORS } from "../theme";

// A large yellow storage bin (the "box"). Simple 2.5D: front face + rim + lip.
export const LegoBin: React.FC<{
  width?: number;
  height?: number;
  open?: number; // 0..1 how open/inviting the mouth reads
  style?: React.CSSProperties;
}> = ({ width = 620, height = 380, open = 1, style }) => {
  const rim = 34;
  return (
    <div style={{ position: "relative", width, height, ...style }}>
      {/* body */}
      <div
        style={{
          position: "absolute",
          inset: `${rim}px 0 0 0`,
          background: `linear-gradient(180deg, ${COLORS.yellowDeep}, #B98E00)`,
          borderRadius: "10px 10px 26px 26px",
          boxShadow: "inset 0 -30px 60px rgba(0,0,0,0.45)",
        }}
      />
      {/* inner mouth (darkness) */}
      <div
        style={{
          position: "absolute",
          left: width * 0.06,
          top: rim * 0.4,
          width: width * 0.88,
          height: rim * 1.7,
          background: `radial-gradient(ellipse at center, #000 30%, #241c00 100%)`,
          borderRadius: "50%",
          opacity: 0.6 + open * 0.4,
        }}
      />
      {/* rim */}
      <div
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          width,
          height: rim * 1.6,
          background: `linear-gradient(180deg, ${COLORS.yellow}, ${COLORS.yellowDeep})`,
          borderRadius: "50%/70%",
          boxShadow: "0 6px 0 rgba(0,0,0,0.25)",
        }}
      />
      {/* stud ridges on the front */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: height * 0.14,
          height: 6,
          background: "rgba(0,0,0,0.18)",
        }}
      />
    </div>
  );
};
