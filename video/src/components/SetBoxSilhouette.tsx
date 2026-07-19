import React from "react";
import { COLORS, FONT } from "../theme";
import { LegoBrick } from "./LegoBrick";

// An attractive abstract LEGO-set-box silhouette used when a price card has no
// real image. Deliberately not the LEGO logo — a generic product box form.
export const SetBoxSilhouette: React.FC<{
  width?: number;
  height?: number;
  accent?: string;
  name: string;
}> = ({ width = 300, height = 380, accent = COLORS.brickRed, name }) => {
  return (
    <div
      style={{
        width,
        height,
        borderRadius: 14,
        background: `linear-gradient(160deg, #17171a, #0c0c0e)`,
        border: `1px solid rgba(255,255,255,0.06)`,
        position: "relative",
        overflow: "hidden",
        boxShadow: "0 30px 60px rgba(0,0,0,0.5)",
      }}
    >
      {/* top brand bar */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: height * 0.16,
          background: accent,
          opacity: 0.9,
        }}
      />
      <div
        style={{
          position: "absolute",
          top: height * 0.05,
          left: width * 0.08,
          fontFamily: FONT,
          fontWeight: 800,
          letterSpacing: "0.06em",
          fontSize: width * 0.075,
          color: "#0A0A0A",
        }}
      >
        {name}
      </div>
      {/* hero build silhouette */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: height * 0.28,
          display: "flex",
          justifyContent: "center",
          transform: "scale(0.9)",
        }}
      >
        <div style={{ position: "relative" }}>
          <div style={{ position: "absolute", left: -30, top: 40 }}>
            <LegoBrick studs={[2, 2]} unit={26} color={accent} />
          </div>
          <div style={{ position: "absolute", left: 40, top: 10 }}>
            <LegoBrick studs={[2, 3]} unit={26} color={COLORS.brickWhite} />
          </div>
          <div style={{ position: "absolute", left: 6, top: 96 }}>
            <LegoBrick studs={[4, 1]} unit={26} color={COLORS.brickYellow} />
          </div>
        </div>
      </div>
      {/* age band / bottom detail */}
      <div
        style={{
          position: "absolute",
          bottom: height * 0.08,
          left: width * 0.08,
          width: width * 0.24,
          height: width * 0.16,
          borderRadius: "50%",
          border: `2px solid ${COLORS.offWhite}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: COLORS.offWhite,
          fontFamily: FONT,
          fontWeight: 800,
          fontSize: width * 0.09,
        }}
      >
        12+
      </div>
    </div>
  );
};
