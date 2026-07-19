import React from "react";
import { useCurrentFrame } from "remotion";
import { COLORS, TYPE } from "../theme";
import { appear, ramp } from "../lib/anim";

// A large, restrained statement. Lines rise and fade. Keep to ~2 lines.
export interface BigStatementProps {
  lines: string[];
  start?: number; // local frame
  inDur?: number;
  hold?: number;
  outDur?: number;
  size?: number;
  align?: "left" | "center";
  accentLineIndex?: number; // render this line in yellow
  weight?: number;
  style?: React.CSSProperties;
}

export const BigStatement: React.FC<BigStatementProps> = ({
  lines,
  start = 0,
  inDur = 16,
  hold = 60,
  outDur = 16,
  size = 92,
  align = "center",
  accentLineIndex,
  weight = 800,
  style,
}) => {
  const frame = useCurrentFrame();
  const globalOpacity = appear(frame, start, inDur, hold, outDur);
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: align === "center" ? "center" : "flex-start",
        textAlign: align,
        opacity: globalOpacity,
        ...style,
      }}
    >
      {lines.map((line, i) => {
        const t = ramp(frame, start + i * 5, start + i * 5 + inDur);
        return (
          <div key={i} style={{ overflow: "hidden", padding: "0 0.04em" }}>
            <div
              style={{
                ...TYPE.statement,
                fontWeight: weight,
                fontSize: size,
                color:
                  accentLineIndex === i ? COLORS.yellow : TYPE.statement.color,
                transform: `translateY(${(1 - t) * 0.7 * size}px)`,
                opacity: t,
              }}
            >
              {line}
            </div>
          </div>
        );
      })}
    </div>
  );
};
