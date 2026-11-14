import type { ArchitectureReport } from "../../api/types";
import { classNames, fmtInt, shape } from "../../lib/format";
import { CheckIcon, Mono } from "../ui/Misc";

export function LayerTable({ report, selected, onSelect }: { report: ArchitectureReport; selected: string | null; onSelect: (n: string | null) => void }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-[13px] tabular">
        <thead className="bg-subtle text-left text-ink-subtle">
          <tr>
            <th className="px-3 py-2 font-bold whitespace-nowrap">Layer</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap">Type</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap">Input</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap">Output</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap">Kernel</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap">Stride</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap">Padding</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap text-right">Groups</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap text-right">Params</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap text-right">Trainable</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap text-right">MACs</th>
            <th className="px-2.5 py-2 font-bold whitespace-nowrap text-right">RF</th>
            <th className="px-2.5 py-2 pr-3 font-bold whitespace-nowrap text-center" />
          </tr>
        </thead>
        <tbody>
          {report.layers.map((s) => (
            <tr
              key={s.name}
              onClick={() => onSelect(selected === s.name ? null : s.name)}
              className={classNames("cursor-pointer border-t border-line transition-colors hover:bg-surface", selected === s.name && "bg-primary-soft/50", !s.valid && "bg-danger-soft/50")}
            >
              <td className="px-3 py-1.5 font-bold whitespace-nowrap text-ink">{s.label}</td>
              <td className="px-2.5 py-1.5 whitespace-nowrap text-ink-muted">{s.type}</td>
              <td className="px-2.5 py-1.5 whitespace-nowrap"><Mono>{shape(s.in_shape)}</Mono></td>
              <td className="px-2.5 py-1.5 whitespace-nowrap"><Mono className="text-link">{shape(s.out_shape)}</Mono></td>
              <td className="px-2.5 py-1.5 whitespace-nowrap"><Mono>{s.kernel ? s.kernel.join("×") : "—"}</Mono></td>
              <td className="px-2.5 py-1.5 whitespace-nowrap"><Mono>{s.stride ? s.stride.join("×") : "—"}</Mono></td>
              <td className="px-2.5 py-1.5 whitespace-nowrap"><Mono>{s.padding && (s.padding[0] || s.padding[1]) ? `${s.padding[0]} | ${s.padding[1]}` : s.padding ? "0" : "—"}</Mono></td>
              <td className="px-2.5 py-1.5 text-right tabular whitespace-nowrap">{s.groups ?? "—"}</td>
              <td className="px-2.5 py-1.5 text-right tabular whitespace-nowrap">{fmtInt(s.params)}</td>
              <td className="px-2.5 py-1.5 text-right tabular whitespace-nowrap">{fmtInt(s.trainable)}</td>
              <td className="px-2.5 py-1.5 text-right tabular whitespace-nowrap">{s.macs ? fmtInt(s.macs) : "—"}</td>
              <td className="px-2.5 py-1.5 text-right tabular whitespace-nowrap">{s.rf_time}</td>
              <td className="px-2.5 py-1.5 pr-3 text-center whitespace-nowrap"><CheckIcon status={s.valid} /></td>
            </tr>
          ))}
        </tbody>
        <tfoot className="border-t border-line-strong bg-surface font-bold">
          <tr>
            <td className="px-3 py-2" colSpan={7}>Total</td>
            <td className="px-2.5 py-2 text-right" />
            <td className="px-2.5 py-2 text-right tabular">{fmtInt(report.total_params)}</td>
            <td className="px-2.5 py-2 text-right tabular">{fmtInt(report.trainable_params)}</td>
            <td className="px-2.5 py-2 text-right tabular">{fmtInt(report.total_macs)}</td>
            <td className="px-2.5 py-2 text-right tabular">{report.receptive_field_samples ?? "—"}</td>
            <td className="px-2.5 py-2 pr-3" />
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
