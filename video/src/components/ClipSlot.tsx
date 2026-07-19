import React from "react";
import { OffthreadVideo, staticFile, useCurrentFrame } from "remotion";
import { COLORS, FONT } from "../theme";
import { ramp } from "../lib/anim";
import { COPY } from "../demoConfig";

// A reusable presentation frame that holds either a REAL clip or an animated
// fallback. The framing, crop, entrance, exit, and overlay behaviour are
// identical, so swapping the mp4 in never requires redesigning the scene.
export interface ClipSlotProps {
  useReal: boolean;
  src: string; // public-relative path
  fallback: React.ReactNode; // intentional animated walkthrough
  overlay?: React.ReactNode; // small status labels drawn on top of either
  start?: number; // local frame the slot appears
  width?: number;
  height?: number;
  muted?: boolean;
  style?: React.CSSProperties;
}

const CornerTicks: React.FC<{ size?: number; color?: string }> = ({
  size = 26,
  color = COLORS.yellow,
}) => (
  <>
    {[
      { top: -2, left: -2, rot: 0 },
      { top: -2, right: -2, rot: 90 },
      { bottom: -2, right: -2, rot: 180 },
      { bottom: -2, left: -2, rot: 270 },
    ].map((c, i) => (
      <div
        key={i}
        style={{
          position: "absolute",
          width: size,
          height: size,
          borderTop: `3px solid ${color}`,
          borderLeft: `3px solid ${color}`,
          transform: `rotate(${c.rot}deg)`,
          ...c,
        }}
      />
    ))}
  </>
);

export const ClipSlot: React.FC<ClipSlotProps> = ({
  useReal,
  src,
  fallback,
  overlay,
  start = 0,
  width = 1440,
  height = 690,
  muted = true,
  style,
}) => {
  const frame = useCurrentFrame();
  const enter = ramp(frame, start, start + 20);
  return (
    <div
      style={{
        position: "relative",
        width,
        height,
        transform: `translateY(${(1 - enter) * 40}px) scale(${
          0.97 + enter * 0.03
        })`,
        opacity: enter,
        ...style,
      }}
    >
      {/* outer frame */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: 10,
          overflow: "hidden",
          background: "#000",
          border: "1px solid rgba(255,255,255,0.10)",
          boxShadow: "0 40px 90px rgba(0,0,0,0.6)",
        }}
      >
        {useReal ? (
          <OffthreadVideo
            src={staticFile(src)}
            muted={muted}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <div style={{ position: "absolute", inset: 0 }}>{fallback}</div>
        )}
        {/* subtle inner vignette for cohesion */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            boxShadow: "inset 0 0 120px rgba(0,0,0,0.5)",
            pointerEvents: "none",
          }}
        />
        {overlay}
      </div>
      <CornerTicks />
      {/* honest fallback marker — never a broken-asset placeholder */}
      {!useReal && (
        <div
          style={{
            position: "absolute",
            bottom: 14,
            right: 16,
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontFamily: FONT,
            fontSize: 13,
            letterSpacing: "0.22em",
            fontWeight: 600,
            color: "rgba(244,241,232,0.55)",
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: COLORS.yellow,
              boxShadow: `0 0 10px ${COLORS.yellow}`,
            }}
          />
          {COPY.systemWalkthrough}
        </div>
      )}
    </div>
  );
};
