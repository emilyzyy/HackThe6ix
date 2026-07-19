import React from "react";
import { COLORS, FONT } from "../theme";

// PLAN -> BUILD -> REPAIR indicator. Active stage yellow, others muted.
export const AgentProgress: React.FC<{
  stages: string[];
  activeIndex: number;
  style?: React.CSSProperties;
}> = ({ stages, activeIndex, style }) => {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 18,
        ...style,
      }}
    >
      {stages.map((stage, i) => {
        const active = i === activeIndex;
        const done = i < activeIndex;
        return (
          <React.Fragment key={stage}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
              }}
            >
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 2,
                  background: active
                    ? COLORS.yellow
                    : done
                    ? "rgba(255,213,0,0.4)"
                    : "rgba(255,255,255,0.12)",
                  boxShadow: active ? `0 0 14px ${COLORS.yellow}` : undefined,
                }}
              />
              <span
                style={{
                  fontFamily: FONT,
                  fontWeight: 700,
                  fontSize: 20,
                  letterSpacing: "0.18em",
                  color: active
                    ? COLORS.yellow
                    : done
                    ? "rgba(244,241,232,0.7)"
                    : COLORS.grayDim,
                }}
              >
                {stage}
              </span>
            </div>
            {i < stages.length - 1 && (
              <div
                style={{
                  width: 40,
                  height: 2,
                  background:
                    i < activeIndex
                      ? "rgba(255,213,0,0.5)"
                      : "rgba(255,255,255,0.12)",
                }}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
