import React from "react";

// A well-designed 2.5D LEGO-like brick: top face with studs, plus front and
// side bevels for depth. No 3D engine — pure CSS. `studs` = [columns, rows]
// looking at the top face. `unit` = pixel size of one stud cell.
export interface LegoBrickProps {
  studs?: [number, number];
  unit?: number;
  color?: string;
  depth?: number; // height of the front bevel (brick vs plate)
  showStuds?: boolean;
  style?: React.CSSProperties;
  glow?: number; // 0..1 yellow discovery glow
  faded?: number; // 0..1 desaturate/dim
}

const shade = (hex: string, amt: number) => {
  const h = hex.replace("#", "");
  const n = parseInt(
    h.length === 3
      ? h
          .split("")
          .map((c) => c + c)
          .join("")
      : h,
    16
  );
  let r = (n >> 16) & 255;
  let g = (n >> 8) & 255;
  let b = n & 255;
  r = Math.max(0, Math.min(255, Math.round(r + amt)));
  g = Math.max(0, Math.min(255, Math.round(g + amt)));
  b = Math.max(0, Math.min(255, Math.round(b + amt)));
  return `rgb(${r},${g},${b})`;
};

export const LegoBrick: React.FC<LegoBrickProps> = ({
  studs = [2, 4],
  unit = 40,
  color = "#C91A09",
  depth,
  showStuds = true,
  style,
  glow = 0,
  faded = 0,
}) => {
  const [cols, rows] = studs;
  const w = cols * unit;
  const h = rows * unit;
  const d = depth ?? Math.round(unit * 0.55);
  const top = shade(color, 18);
  const front = shade(color, -46);
  const side = shade(color, -74);
  const studTop = shade(color, 30);

  return (
    <div
      style={{
        position: "relative",
        width: w,
        height: h + d,
        filter: faded ? `saturate(${1 - faded * 0.85}) brightness(${1 - faded * 0.45})` : undefined,
        ...style,
      }}
    >
      {/* front bevel */}
      <div
        style={{
          position: "absolute",
          left: 0,
          top: h - d * 0.15,
          width: w,
          height: d,
          background: `linear-gradient(${front}, ${side})`,
          borderRadius: `0 0 ${unit * 0.12}px ${unit * 0.12}px`,
        }}
      />
      {/* top face */}
      <div
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          width: w,
          height: h,
          background: `linear-gradient(135deg, ${top}, ${color})`,
          borderRadius: unit * 0.1,
          boxShadow: `inset 0 1px 0 ${shade(color, 40)}, inset 0 -2px 6px ${shade(
            color,
            -30
          )}`,
        }}
      >
        {showStuds &&
          Array.from({ length: rows }).map((_, ri) =>
            Array.from({ length: cols }).map((__, ci) => (
              <div
                key={`${ri}-${ci}`}
                style={{
                  position: "absolute",
                  left: ci * unit + unit * 0.2,
                  top: ri * unit + unit * 0.2,
                  width: unit * 0.6,
                  height: unit * 0.6,
                  borderRadius: "50%",
                  background: `radial-gradient(circle at 38% 32%, ${studTop}, ${color} 62%, ${shade(
                    color,
                    -28
                  )})`,
                  boxShadow: `0 ${unit * 0.03}px ${unit * 0.06}px ${shade(
                    color,
                    -60
                  )}`,
                }}
              />
            ))
          )}
      </div>
      {/* discovery glow */}
      {glow > 0 && (
        <div
          style={{
            position: "absolute",
            inset: -unit * 0.35,
            borderRadius: unit * 0.3,
            boxShadow: `0 0 ${unit * 0.9}px ${unit * 0.4}px rgba(255,213,0,${
              0.55 * glow
            })`,
            border: `2px solid rgba(255,213,0,${0.9 * glow})`,
            pointerEvents: "none",
          }}
        />
      )}
    </div>
  );
};
