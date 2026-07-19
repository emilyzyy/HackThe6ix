import React from "react";
import { COLORS, FONT } from "../theme";

// A floating camera viewport frame. Holds a mini-view (children) and reads as a
// physical capture window moving through space.
export const CameraFrame: React.FC<{
  width?: number;
  height?: number;
  label?: string;
  active?: boolean;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ width = 300, height = 200, label, active = false, children, style }) => {
  const c = active ? COLORS.yellow : "rgba(255,255,255,0.28)";
  return (
    <div
      style={{
        position: "relative",
        width,
        height,
        border: `1.5px solid ${c}`,
        borderRadius: 6,
        background: "rgba(10,10,12,0.65)",
        overflow: "hidden",
        boxShadow: active
          ? `0 0 30px rgba(255,213,0,0.25)`
          : "0 12px 40px rgba(0,0,0,0.5)",
        ...style,
      }}
    >
      <div style={{ position: "absolute", inset: 0 }}>{children}</div>
      {/* corner ticks */}
      {[
        { top: 6, left: 6, b: "top-left" },
        { top: 6, right: 6, b: "top-right" },
        { bottom: 6, left: 6, b: "bottom-left" },
        { bottom: 6, right: 6, b: "bottom-right" },
      ].map((p, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            width: 10,
            height: 10,
            borderTop: i < 2 ? `2px solid ${c}` : undefined,
            borderBottom: i >= 2 ? `2px solid ${c}` : undefined,
            borderLeft: i % 2 === 0 ? `2px solid ${c}` : undefined,
            borderRight: i % 2 === 1 ? `2px solid ${c}` : undefined,
            top: p.top,
            left: p.left,
            right: p.right,
            bottom: p.bottom,
          }}
        />
      ))}
      {label && (
        <div
          style={{
            position: "absolute",
            top: 8,
            left: 10,
            fontFamily: FONT,
            fontSize: 12,
            letterSpacing: "0.18em",
            fontWeight: 600,
            color: active ? COLORS.yellow : "rgba(244,241,232,0.6)",
          }}
        >
          {label}
        </div>
      )}
      {/* recording dot */}
      <div
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: active ? COLORS.yellow : "rgba(255,255,255,0.4)",
        }}
      />
    </div>
  );
};
