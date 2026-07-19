import React from "react";
import { useCurrentFrame } from "remotion";
import { COLORS, TYPE } from "../theme";
import { appear } from "../lib/anim";

// Small uppercase technical caption with an optional yellow tick marker.
export interface TechnicalLabelProps {
  text: string;
  start?: number;
  inDur?: number;
  hold?: number;
  outDur?: number;
  size?: number;
  color?: string;
  marker?: boolean;
  active?: boolean; // yellow when active
  style?: React.CSSProperties;
}

export const TechnicalLabel: React.FC<TechnicalLabelProps> = ({
  text,
  start = 0,
  inDur = 10,
  hold = 40,
  outDur = 10,
  size = 20,
  color,
  marker = true,
  active = false,
  style,
}) => {
  const frame = useCurrentFrame();
  const opacity = appear(frame, start, inDur, hold, outDur);
  const c = color ?? (active ? COLORS.yellow : COLORS.gray);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: size * 0.6,
        opacity,
        ...style,
      }}
    >
      {marker && (
        <div
          style={{
            width: size * 0.5,
            height: size * 0.5,
            background: active ? COLORS.yellow : COLORS.grayDim,
            boxShadow: active
              ? `0 0 ${size}px ${COLORS.yellow}`
              : undefined,
          }}
        />
      )}
      <span
        style={{
          ...TYPE.technical,
          fontSize: size,
          color: c,
        }}
      >
        {text}
      </span>
    </div>
  );
};
