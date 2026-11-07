import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { useUI } from "../../store/ui";

echarts.use([
  LineChart, BarChart, HeatmapChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, VisualMapComponent,
  MarkAreaComponent, MarkLineComponent, MarkPointComponent, DataZoomComponent, ToolboxComponent, TitleComponent, CanvasRenderer,
]);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Option = Record<string, any>;

export interface ChartTokens {
  ink: string;
  muted: string;
  subtle: string;
  line: string;
  card: string;
  surface: string;
  font: string;
  mono: string;
  palette: string[];
  heatLow: string;
  heatHigh: string;
  divNeg: string;
  divMid: string;
  divPos: string;
  primary: string;
  onPrimary: string;
  danger: string;
  success: string;
  warning: string;
}

function readTokens(): ChartTokens {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string, fb: string) => cs.getPropertyValue(name).trim() || fb;
  return {
    ink: v("--c-ink", "#1b1b1f"),
    muted: v("--c-ink-muted", "#4a4a50"),
    subtle: v("--c-ink-subtle", "#6f6e73"),
    line: v("--c-line", "#e4e1da"),
    card: v("--c-card", "#ffffff"),
    surface: v("--c-surface", "#fbfaf7"),
    font: '"Atkinson Hyperlegible Next Variable", system-ui, sans-serif',
    mono: '"Atkinson Hyperlegible Mono Variable", ui-monospace, monospace',
    palette: [1, 2, 3, 4, 5, 6, 7].map((i) => v(`--chart-${i}`, ["#1f2b6c", "#b45309", "#1e7a46", "#b42318", "#6d4aae", "#0f766e", "#64748b"][i - 1])),
    heatLow: v("--heat-low", "#f6f4ef"),
    heatHigh: v("--heat-high", "#1f2b6c"),
    divNeg: v("--div-neg", "#b42318"),
    divMid: v("--div-mid", "#faf9f6"),
    divPos: v("--div-pos", "#1f2b6c"),
    primary: v("--c-primary", "#1f2b6c"),
    onPrimary: v("--c-on-primary", "#ffffff"),
    danger: v("--c-danger", "#b42318"),
    success: v("--c-success", "#1e7a46"),
    warning: v("--c-warning", "#8a5300"),
  };
}

export function useChartTokens(): ChartTokens {
  const theme = useUI((s) => s.resolvedTheme);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => readTokens(), [theme]);
}

export function baseOption(t: ChartTokens, extra: Option = {}): Option {
  return {
    color: t.palette,
    textStyle: { fontFamily: t.font, color: t.muted },
    animationDuration: 250,
    tooltip: {
      backgroundColor: t.card,
      borderColor: t.line,
      textStyle: { color: t.ink, fontFamily: t.font, fontSize: 12.5 },
      extraCssText: "border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.12);",
    },
    toolbox: {
      right: 4,
      top: 0,
      itemSize: 13,
      iconStyle: { borderColor: t.subtle },
      feature: { saveAsImage: { title: "Save as PNG", pixelRatio: 2, backgroundColor: t.card } },
    },
    ...extra,
  };
}

export function axis(t: ChartTokens, extra: Option = {}): Option {
  return {
    axisLine: { lineStyle: { color: t.line } },
    axisTick: { show: false },
    axisLabel: { color: t.subtle, fontSize: 11, fontFamily: t.font, formatter: (v: number | string) => (typeof v === "number" && !Number.isInteger(v) ? String(Number(v.toPrecision(3))) : String(v)) },
    splitLine: { lineStyle: { color: t.line, type: "dashed" } },
    nameTextStyle: { color: t.subtle, fontSize: 11, fontFamily: t.font },
    nameGap: 26,
    ...extra,
  };
}

export function EChart({
  option,
  height = 260,
  ariaLabel,
  onClick,
}: {
  option: Option;
  height?: number;
  ariaLabel: string;
  onClick?: (params: Option) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;

  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    let chart: echarts.ECharts;
    try {
      chart = echarts.init(ref.current, undefined, { renderer: "canvas" });
    } catch {
      return; // non-browser environment (tests)
    }
    inst.current = chart;
    chart.on("click", (p) => clickRef.current?.(p as Option));
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      chart.dispose();
      inst.current = null;
    };
  }, []);

  useEffect(() => {
    inst.current?.setOption(option, { notMerge: true, lazyUpdate: true });
  }, [option]);

  return <div ref={ref} role="img" aria-label={ariaLabel} style={{ height, width: "100%" }} />;
}
