import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { BigStatement } from "../components/BigStatement";
import { TechnicalLabel } from "../components/TechnicalLabel";
import { InventoryRow } from "../components/InventoryRow";
import { LegoBrick } from "../components/LegoBrick";
import { COLORS } from "../theme";
import { ramp } from "../lib/anim";
import { inventoryRows, COPY } from "../demoConfig";

export const InventoryReveal: React.FC = () => {
  const frame = useCurrentFrame();
  const outFade = ramp(420 - frame, 0, 14);
  const zoomOut = ramp(frame, 250, 320);
  const dropBrick = ramp(frame, 350, 420);

  return (
    <AbsoluteFill style={{ opacity: 1 }}>
      <Backdrop />

      {/* the inventory assembles */}
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 14,
            transform: `scale(${1 - zoomOut * 0.14}) translateY(${zoomOut * -10}px)`,
          }}
        >
          <div style={{ marginBottom: 10 }}>
            <TechnicalLabel text="INVENTORY" start={10} inDur={12} hold={9999} outDur={0} size={18} active />
          </div>
          {inventoryRows.map((row, i) => (
            <InventoryRow
              key={row.part}
              studs={row.studs}
              colorHex={row.colorHex}
              part={row.part}
              color={row.color}
              qty={row.qty}
              start={6 + i * 11}
            />
          ))}
        </div>
      </AbsoluteFill>

      {/* text progression */}
      <div style={{ position: "absolute", top: 70, width: "100%", display: "flex", justifyContent: "center" }}>
        <BigStatement lines={[COPY.inventory.fromA]} start={160} inDur={12} hold={40} outDur={14} size={40} />
      </div>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <BigStatement lines={[COPY.inventory.toA, COPY.inventory.toB]} start={230} inDur={14} hold={70} outDur={16} size={66} accentLineIndex={1} style={{ opacity: zoomOut }} />
      </AbsoluteFill>
      <div style={{ position: "absolute", bottom: 120, width: "100%", display: "flex", justifyContent: "center" }}>
        <TechnicalLabel text={COPY.inventory.ready} start={330} inDur={12} hold={70} outDur={14} size={26} active />
      </div>

      {/* a single yellow brick drops through the bottom of frame */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: 300 + dropBrick * 900,
          transform: `translateX(-50%) rotate(${dropBrick * 120}deg)`,
          opacity: dropBrick > 0.05 ? 1 : 0,
        }}
      >
        <LegoBrick studs={[2, 2]} unit={46} color={COLORS.yellow} glow={0.4} />
      </div>
    </AbsoluteFill>
  );
};
