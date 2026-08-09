"use client";

import dynamic from "next/dynamic";
import { useMemo, useRef } from "react";

// Plotly touches `window` at import time, so it must never be pulled into the
// server bundle — dynamic import with ssr:false is the standard escape hatch.
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export type PlotlyFigure = {
  data: unknown[];
  layout?: Record<string, unknown>;
};

export default function PlotlyChart({
  figure,
  filenameForExport = "chart",
  className = "",
}: {
  figure: PlotlyFigure;
  filenameForExport?: string;
  className?: string;
}) {
  const divRef = useRef<HTMLDivElement>(null);

  const layout = useMemo(
    () => ({
      autosize: true,
      ...figure.layout,
    }),
    [figure.layout]
  );

  return (
    <div ref={divRef} className={className}>
      <Plot
        data={figure.data as never}
        layout={layout as never}
        config={{
          responsive: true,
          displaylogo: false,
          toImageButtonOptions: { format: "png", filename: filenameForExport, scale: 2 },
          modeBarButtonsToAdd: [],
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
