import { useMemo } from "react";
import { axis, baseOption, EChart, useChartTokens, type Option } from "./EChart";

export interface LineSeries {
  name: string;
  data: (number | null)[];
  color?: string;
  dashed?: boolean;
  area?: boolean;
  width?: number;
  step?: boolean;
}

export function LinePlot({
  x,
  series,
  xName,
  yName,
  height = 240,
  ariaLabel,
  markAreaX,
  markLineX,
  yMin,
  yMax,
  legend = true,
  xIsCategory,
  zoom,
}: {
  x: (number | string)[];
  series: LineSeries[];
  xName?: string;
  yName?: string;
  height?: number;
  ariaLabel: string;
  markAreaX?: [number, number] | null;
  markLineX?: number | null;
  yMin?: number;
  yMax?: number;
  legend?: boolean;
  xIsCategory?: boolean;
  zoom?: boolean;
}) {
  const t = useChartTokens();
  const option = useMemo<Option>(() => {
    const cat = xIsCategory ?? false;
    return baseOption(t, {
      grid: { left: 54, right: 18, top: legend ? 34 : 18, bottom: zoom ? 58 : 40 },
      legend: legend ? { top: 0, left: 0, textStyle: { color: t.muted, fontSize: 12 }, icon: "roundRect", itemHeight: 4, itemWidth: 14 } : undefined,
      tooltip: { ...baseOption(t).tooltip, trigger: "axis", valueFormatter: (v: number) => (typeof v === "number" ? v.toPrecision(4) : v) },
      xAxis: axis(t, {
        type: cat ? "category" : "value",
        data: cat ? x : undefined,
        name: xName,
        nameLocation: "middle",
        min: cat ? undefined : "dataMin",
        max: cat ? undefined : "dataMax",
        splitLine: { show: false },
      }),
      yAxis: axis(t, { type: "value", name: yName, nameLocation: "middle", nameGap: 40, min: yMin, max: yMax, scale: yMin === undefined }),
      dataZoom: zoom ? [{ type: "inside" }, { type: "slider", height: 16, bottom: 8, borderColor: t.line, textStyle: { color: t.subtle } }] : undefined,
      series: series.map((s, i) => ({
        name: s.name,
        type: "line",
        showSymbol: false,
        step: s.step ? "middle" : undefined,
        data: cat ? s.data : s.data.map((v, j) => [x[j], v]),
        lineStyle: { width: s.width ?? 1.6, type: s.dashed ? "dashed" : "solid", color: s.color },
        itemStyle: { color: s.color },
        areaStyle: s.area ? { opacity: 0.12 } : undefined,
        markArea:
          i === 0 && markAreaX
            ? { silent: true, itemStyle: { color: t.warning, opacity: 0.13 }, data: [[{ xAxis: markAreaX[0] }, { xAxis: markAreaX[1] }]] }
            : undefined,
        markLine:
          i === 0 && markLineX !== null && markLineX !== undefined
            ? { silent: true, symbol: "none", lineStyle: { color: t.danger, type: "solid" }, label: { show: false }, data: [{ xAxis: markLineX }] }
            : undefined,
      })),
    });
  }, [t, x, series, xName, yName, legend, markAreaX, markLineX, yMin, yMax, xIsCategory, zoom]);
  return <EChart option={option} height={height} ariaLabel={ariaLabel} />;
}

/** Multi-channel signal display with per-channel vertical offsets. */
export function StackedSignals({
  times,
  data,
  channels,
  unit,
  height,
  ariaLabel,
  highlight,
  compare,
  compareLabel,
}: {
  times: number[];
  data: number[][];
  channels: string[];
  unit: string;
  height?: number;
  ariaLabel: string;
  highlight?: [number, number] | null;
  compare?: number[][] | null;
  compareLabel?: string;
}) {
  const t = useChartTokens();
  const option = useMemo<Option>(() => {
    const all = data.flat().concat(compare ? compare.flat() : []);
    const sorted = all.map(Math.abs).sort((a, b) => a - b);
    const p95 = sorted[Math.floor(sorted.length * 0.95)] || 1;
    const spacing = 2.2 * p95 || 1;
    const n = data.length;
    const mk = (rows: number[][], name: string, color: string, width: number) =>
      rows.map((row, i) => ({
        name: i === 0 ? name : `${name}-${i}`,
        type: "line",
        showSymbol: false,
        silent: true,
        data: row.map((v, j) => [times[j], v + (n - 1 - i) * spacing]),
        lineStyle: { width, color },
        markArea:
          i === 0 && highlight && name !== compareLabel
            ? { silent: true, itemStyle: { color: t.warning, opacity: 0.12 }, data: [[{ xAxis: highlight[0] }, { xAxis: highlight[1] }]] }
            : undefined,
      }));
    const series = [...(compare ? mk(compare, compareLabel ?? "before", t.subtle, 1) : []), ...mk(data, "signal", t.palette[0], 1.2)];
    return baseOption(t, {
      grid: { left: 64, right: 16, top: 16, bottom: 40 },
      tooltip: { show: false },
      xAxis: axis(t, { type: "value", min: times[0], max: times[times.length - 1], name: "time (s)", nameLocation: "middle", splitLine: { show: false } }),
      yAxis: axis(t, {
        type: "value",
        min: -spacing,
        max: n * spacing,
        interval: spacing,
        axisLabel: {
          color: t.subtle,
          fontSize: 11,
          formatter: (v: number) => {
            const k = Math.round(v / spacing);
            return Math.abs(v - k * spacing) < spacing * 0.01 && k >= 0 && k < n ? channels[n - 1 - k] : "";
          },
        },
        splitLine: { show: false },
      }),
      graphic: undefined,
      series,
      title: { text: `scale: ${spacing.toPrecision(2)} ${unit} / row`, right: 34, top: 0, textStyle: { fontSize: 11, color: t.subtle, fontWeight: "normal" } },
    });
  }, [t, times, data, channels, unit, highlight, compare, compareLabel]);
  return <EChart option={option} height={height ?? Math.max(180, 26 * data.length + 70)} ariaLabel={ariaLabel} />;
}

export function Heatmap({
  matrix,
  xLabels,
  yLabels,
  diverging,
  height = 260,
  ariaLabel,
  xName,
  yName,
  highlightCol,
  highlightRow,
  showValues,
  onCellClick,
  valueDigits = 3,
}: {
  matrix: number[][];
  xLabels?: (string | number)[];
  yLabels?: (string | number)[];
  diverging?: boolean;
  height?: number;
  ariaLabel: string;
  xName?: string;
  yName?: string;
  highlightCol?: number | null;
  highlightRow?: number | null;
  showValues?: boolean;
  onCellClick?: (row: number, col: number) => void;
  valueDigits?: number;
}) {
  const t = useChartTokens();
  const option = useMemo<Option>(() => {
    const rows = matrix.length;
    const cols = rows ? matrix[0].length : 0;
    const data: [number, number, number][] = [];
    let lo = Infinity;
    let hi = -Infinity;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const v = matrix[r][c];
        data.push([c, r, v]);
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
    const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    const xl = xLabels ?? Array.from({ length: cols }, (_, i) => i);
    const yl = yLabels ?? Array.from({ length: rows }, (_, i) => i);
    return baseOption(t, {
      grid: { left: 70, right: 70, top: 14, bottom: 44 },
      tooltip: {
        ...baseOption(t).tooltip,
        formatter: (p: Option) => `${yl[p.value[1]]} · ${xl[p.value[0]]}<br/><b>${Number(p.value[2]).toPrecision(4)}</b>`,
      },
      xAxis: axis(t, { type: "category", data: xl, name: xName, nameLocation: "middle", splitArea: { show: false }, axisLabel: { color: t.subtle, fontSize: 10, hideOverlap: true } }),
      yAxis: axis(t, { type: "category", data: yl, name: yName, nameLocation: "middle", nameGap: 52, inverse: true, axisLabel: { color: t.subtle, fontSize: 10, hideOverlap: true } }),
      visualMap: {
        min: diverging ? -m : lo,
        max: diverging ? m : hi === lo ? lo + 1 : hi,
        calculable: false,
        orient: "vertical",
        right: 4,
        top: "middle",
        itemHeight: Math.max(60, height - 110),
        itemWidth: 10,
        textStyle: { color: t.subtle, fontSize: 10 },
        inRange: { color: diverging ? [t.divNeg, t.divMid, t.divPos] : [t.heatLow, t.heatHigh] },
        formatter: (v: number) => v.toPrecision(2),
      },
      series: [
        {
          type: "heatmap",
          data: showValues
            ? data.map((d) => {
                const lo2 = diverging ? -m : lo;
                const hi2 = diverging ? m : hi;
                const frac = hi2 > lo2 ? (Math.abs(d[2]) - (diverging ? 0 : lo2)) / (hi2 - (diverging ? 0 : lo2)) : 0;
                return { value: d, label: { color: frac > 0.55 ? t.onPrimary : t.ink } };
              })
            : data,
          progressive: 4000,
          label: showValues ? { show: true, fontSize: 11, formatter: (p: Option) => Number(p.value[2]).toFixed(valueDigits) } : undefined,
          emphasis: { itemStyle: { borderColor: t.ink, borderWidth: 1 } },
          markArea:
            highlightCol !== null && highlightCol !== undefined
              ? { silent: true, itemStyle: { color: "transparent", borderColor: t.warning, borderWidth: 2 }, data: [[{ xAxis: highlightCol }, { xAxis: highlightCol }]] }
              : highlightRow !== null && highlightRow !== undefined
                ? { silent: true, itemStyle: { color: "transparent", borderColor: t.warning, borderWidth: 2 }, data: [[{ yAxis: highlightRow }, { yAxis: highlightRow }]] }
                : undefined,
        },
      ],
    });
  }, [t, matrix, xLabels, yLabels, diverging, height, xName, yName, highlightCol, highlightRow, showValues, valueDigits]);
  return (
    <EChart
      option={option}
      height={height}
      ariaLabel={ariaLabel}
      onClick={onCellClick ? (p) => Array.isArray(p.value) && onCellClick(p.value[1] as number, p.value[0] as number) : undefined}
    />
  );
}

export function Bars({
  labels,
  series,
  height = 220,
  ariaLabel,
  yName,
  horizontal,
  highlightIndex,
  yMax,
  stacked,
  valueDigits = 3,
}: {
  labels: (string | number)[];
  series: { name: string; data: (number | null)[]; color?: string }[];
  height?: number;
  ariaLabel: string;
  yName?: string;
  horizontal?: boolean;
  highlightIndex?: number | null;
  yMax?: number;
  stacked?: boolean;
  valueDigits?: number;
}) {
  const t = useChartTokens();
  const option = useMemo<Option>(() => {
    const cat = axis(t, { type: "category", data: labels, splitLine: { show: false }, axisLabel: { color: t.subtle, fontSize: 11, hideOverlap: true, interval: "auto" } });
    const val = axis(t, { type: "value", name: yName, nameLocation: "middle", nameGap: 40, max: yMax });
    return baseOption(t, {
      grid: { left: horizontal ? 96 : 54, right: 18, top: series.length > 1 ? 34 : 18, bottom: 36 },
      legend: series.length > 1 ? { top: 0, left: 0, textStyle: { color: t.muted, fontSize: 12 } } : undefined,
      tooltip: { ...baseOption(t).tooltip, trigger: "axis", valueFormatter: (v: number) => (typeof v === "number" ? v.toFixed(valueDigits) : v) },
      xAxis: horizontal ? val : cat,
      yAxis: horizontal ? { ...cat, inverse: true } : val,
      series: series.map((s, si) => ({
        name: s.name,
        type: "bar",
        stack: stacked ? "s" : undefined,
        barMaxWidth: 28,
        data: s.data.map((v, i) => ({
          value: v,
          itemStyle: {
            color: highlightIndex === i && si === 0 ? t.warning : (s.color ?? t.palette[si % t.palette.length]),
            borderRadius: horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0],
          },
        })),
      })),
    });
  }, [t, labels, series, yName, horizontal, highlightIndex, yMax, stacked, valueDigits]);
  return <EChart option={option} height={height} ariaLabel={ariaLabel} />;
}
