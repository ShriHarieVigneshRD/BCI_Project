import type { ArchitectureReport } from "../../api/types";
import { classNames, fmtInt, shape } from "../../lib/format";
import { CheckIcon, Mono } from "../ui/Misc";

export function LayerTable({ report, selected, onSelect }: { report: ArchitectureReport; selected: string | null; onSelect: (n: string | null) => void }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-[13px] tabular">
        <thead className="bg-subtle text-left text-ink-subtle">
          <tr>
            {["Layer", "Type", "Input", "Output", "Kernel", "Stride", "Padding", "Groups", "Params", "Trainable", "MACs", "RF", ""].map((h) => (
              <th key={h} className="px-2.5 py-2 font-bold whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {report.layers.map((s) => (
            <tr
              key={s.name}
              onClick={() => onSelect(selected === s.name ? null : s.name)}
              className={classNames("cursor-pointer border-t border-line hover:bg-surface", selected === s.name && "bg-primary-soft/50", !s.valid && "bg-danger-soft/50")}
            >
              <td className="px-2.5 py-1.5 font-bold whitespace-nowrap text-ink">{s.label}</td>
              <td className="px-2.5 py-1.5 whitespace-nowrap text-ink-muted">{s.type}</td>
              <td className="px-2.5 py-1.5"><Mono>{shape(s.in_shape)}</Mono></td>
              <td className="px-2.5 py-1.5"><Mono className="text-link">{shape(s.out_shape)}</Mono></td>
              <td className="px-2.5 py-1.5"><Mono>{s.kernel ? s.kernel.join("×") : "—"}</Mono></td>
              <td className="px-2.5 py-1.5"><Mono>{s.stride ? s.stride.join("×") : "—"}</Mono></td>
              <td className="px-2.5 py-1.5"><Mono>{s.padding && (s.padding[0] || s.padding[1]) ? `${s.padding[0]} | ${s.padding[1]}` : s.padding ? "0" : "—"}</Mono></td>
              <td className="px-2.5 py-1.5">{s.groups ?? "—"}</td>
              <td className="px-2.5 py-1.5">{fmtInt(s.params)}</td>
              <td className="px-2.5 py-1.5">{fmtInt(s.trainable)}</td>
              <td className="px-2.5 py-1.5">{s.macs ? fmtInt(s.macs) : "—"}</td>
              <td className="px-2.5 py-1.5">{s.rf_time}</td>
              <td className="px-2.5 py-1.5"><CheckIcon status={s.valid} /></td>
            </tr>
          ))}
        </tbody>
        <tfoot className="border-t border-line-strong bg-surface font-bold">
          <tr>
            <td className="px-2.5 py-2" colSpan={8}>Total</td>
            <td className="px-2.5 py-2">{fmtInt(report.total_params)}</td>
            <td className="px-2.5 py-2">{fmtInt(report.trainable_params)}</td>
            <td className="px-2.5 py-2">{fmtInt(report.total_macs)}</td>
            <td className="px-2.5 py-2" colSpan={2}>{report.receptive_field_samples ?? "—"}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
