import React from "react";
import { AbsoluteFill } from "remotion";
import { COLORS } from "../theme";

// Shared world backdrop: deep black with a faint stud-grid texture and a
// cinematic vignette, so every scene reads as one continuous space.
export const Backdrop: React.FC<{
  studGrid?: boolean;
  tint?: string;
  vignette?: number;
}> = ({ studGrid = true, tint, vignette = 0.9 }) => {
  return (
    <AbsoluteFill style={{ background: COLORS.black }}>
      {tint && (
        <AbsoluteFill style={{ background: tint, opacity: 0.5 }} />
      )}
      {studGrid && (
        <AbsoluteFill
          style={{
            backgroundImage: `radial-gradient(circle at center, rgba(255,213,0,0.05) 2px, transparent 2.4px)`,
            backgroundSize: "72px 72px",
            opacity: 0.5,
            maskImage:
              "radial-gradient(ellipse 80% 70% at 50% 50%, black 40%, transparent 85%)",
            WebkitMaskImage:
              "radial-gradient(ellipse 80% 70% at 50% 50%, black 40%, transparent 85%)",
          }}
        />
      )}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse 75% 65% at 50% 48%, transparent 45%, rgba(0,0,0,${vignette}) 100%)`,
        }}
      />
    </AbsoluteFill>
  );
};
