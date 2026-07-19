import React from "react";
import { COLORS } from "../theme";
import { rand } from "../lib/anim";

// A top-down table map: the fitted yellow plane with a stud grid and sparse
// point-cloud dots. `fit` 0..1 = how stable/clean the plane fit is.
export const TableMap: React.FC<{
  width?: number;
  height?: number;
  fit?: number; // 0 = raw points, 1 = clean fitted plane
  points?: number;
  gridReveal?: number; // 0..1 grid draw-in
  style?: React.CSSProperties;
  children?: React.ReactNode;
}> = ({
  width = 900,
  height = 560,
  fit = 1,
  points = 60,
  gridReveal = 1,
  style,
  children,
}) => {
  return (
    <div
      style={{
        position: "relative",
        width,
        height,
        transformStyle: "preserve-3d",
        ...style,
      }}
    >
      {/* fitted plane */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: 18,
          background: `linear-gradient(180deg, rgba(255,213,0,${
            0.06 * fit
          }), rgba(255,213,0,${0.02 * fit}))`,
          border: `1.5px solid rgba(255,213,0,${0.15 + 0.45 * fit})`,
          boxShadow: `inset 0 0 80px rgba(255,213,0,${0.05 * fit})`,
          opacity: 0.4 + 0.6 * fit,
        }}
      />
      {/* stud grid */}
      <div
        style={{
          position: "absolute",
          inset: 12,
          borderRadius: 12,
          backgroundImage: `radial-gradient(circle, rgba(255,213,0,${
            0.4 * fit
          }) 1.5px, transparent 2px)`,
          backgroundSize: "46px 46px",
          opacity: gridReveal,
          maskImage:
            "linear-gradient(180deg, black, black 80%, transparent)",
          WebkitMaskImage:
            "linear-gradient(180deg, black, black 80%, transparent)",
        }}
      />
      {/* sparse point cloud (fades as fit -> 1) */}
      {Array.from({ length: points }).map((_, i) => {
        const rx = rand(i * 2 + 3);
        const ry = rand(i * 5 + 11);
        const jitter = (1 - fit) * 26;
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: rx * (width - 40) + 20 + (rand(i) - 0.5) * jitter,
              top: ry * (height - 40) + 20 + (rand(i + 1) - 0.5) * jitter,
              width: 3,
              height: 3,
              borderRadius: "50%",
              background: COLORS.yellow,
              opacity: 0.25 + (1 - fit) * 0.5,
            }}
          />
        );
      })}
      {children}
    </div>
  );
};
