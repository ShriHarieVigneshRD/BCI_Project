import type { ArchitectureConfig } from "../../api/types";
import { NumberField, SelectField, Toggle } from "../ui/Field";

export function presetValues(name: string, sfreq: number | null): Partial<ArchitectureConfig> {
  const k1 = sfreq ? Math.max(2, Math.round(sfreq / 2)) : 64;
  const common: Partial<ArchitectureConfig> = {
    temporal_kernel: k1, temporal_stride: 1, spatial_kernel: null, pool1_kernel: 4, pool1_stride: null, separable_kernel: 16,
    pool2_kernel: 8, pool2_stride: null, pool_type: "avg", padding: "same", dropout: 0.5, dropout_type: "dropout",
    activation: "elu", batch_norm: true, bn_momentum: 0.01, bn_eps: 0.001, depthwise_max_norm: 1, classifier_max_norm: 0.25,
  };
  if (name === "eegnet-4,2") return { ...common, preset: name, F1: 4, D: 2, F2: 8 };
  if (name === "eegnet-16,2") return { ...common, preset: name, F1: 16, D: 2, F2: 32 };
  if (name === "eegnet-8,2") return { ...common, preset: name, F1: 8, D: 2, F2: 16 };
  return { preset: "custom" };
}

/** Editable EEGNet parameters, grouped into structural and training-time settings. */
export function ArchForm({
  value,
  onChange,
  base,
  editableDims,
  disabled,
}: {
  value: ArchitectureConfig;
  onChange: (c: ArchitectureConfig) => void;
  base?: ArchitectureConfig | null;
  editableDims?: boolean;
  disabled?: boolean;
}) {
  const set = (patch: Partial<ArchitectureConfig>) => onChange({ ...value, ...patch, preset: "preset" in patch ? (patch.preset as string) : "custom" });
  const ch = (k: keyof ArchitectureConfig) => !!base && base[k] !== value[k];
  const num = (k: keyof ArchitectureConfig, label: string, hint?: string, opts: { nullable?: boolean; placeholder?: string; step?: number; min?: number } = {}) => (
    <NumberField
      label={label}
      hint={hint}
      value={value[k] as number | null}
      nullable={opts.nullable}
      placeholder={opts.placeholder}
      step={opts.step ?? 1}
      min={opts.min}
      disabled={disabled}
      changed={ch(k)}
      onChange={(v) => set({ [k]: v } as Partial<ArchitectureConfig>)}
    />
  );
  return (
    <div className="space-y-6">
      <SelectField
        label="Preset"
        value={value.preset as "eegnet-8,2" | "eegnet-4,2" | "eegnet-16,2" | "custom"}
        disabled={disabled}
        hint="EEGNet-8,2 is the original baseline (Lawhern et al., 2018). Choosing a preset resets the structural parameters."
        onChange={(p) => p !== "custom" && onChange({ ...value, ...presetValues(p, value.sfreq) })}
        options={[
          { value: "eegnet-8,2", label: "EEGNet-8,2 · original baseline" },
          { value: "eegnet-4,2", label: "EEGNet-4,2 · compact" },
          { value: "eegnet-16,2", label: "EEGNet-16,2 · wider" },
          { value: "custom", label: "Custom" },
        ]}
      />
      <fieldset>
        <legend className="mb-3 text-[12px] font-bold uppercase tracking-[0.07em] text-ink-subtle">Input (from the preprocessed data)</legend>
        <div className="grid grid-cols-3 gap-3">
          <NumberField label="Channels C" value={value.n_channels} disabled={!editableDims || disabled} changed={ch("n_channels")} onChange={(v) => set({ n_channels: v ?? 1 })} />
          <NumberField label="Samples T" value={value.n_times} disabled={!editableDims || disabled} changed={ch("n_times")} onChange={(v) => set({ n_times: v ?? 1 })} />
          <NumberField label="Classes" value={value.n_classes} disabled={!editableDims || disabled} changed={ch("n_classes")} onChange={(v) => set({ n_classes: v ?? 2 })} />
        </div>
      </fieldset>
      <fieldset>
        <legend className="mb-1 text-[12px] font-bold uppercase tracking-[0.07em] text-ink-subtle">Structural parameters</legend>
        <p className="mb-3 text-[12.5px] text-ink-subtle">These change the computation graph and the shapes of learned weights.</p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {num("F1", "Temporal filters F1", "Number of temporal (band-pass-like) filters.", { min: 1 })}
          {num("D", "Depth multiplier D", "Spatial filters learned per temporal filter.", { min: 1 })}
          {num("F2", "Pointwise filters F2", "Feature maps after the separable convolution; original EEGNet uses F1×D.", { min: 1 })}
          {num("temporal_kernel", "Temporal kernel", value.sfreq ? `Samples; ≈ sfreq/2 = ${Math.round(value.sfreq / 2)} covers 500 ms.` : "Samples.", { min: 1 })}
          {num("temporal_stride", "Temporal stride", undefined, { min: 1 })}
          {num("spatial_kernel", "Spatial kernel height", "Channels covered by each depthwise spatial filter. Empty = all channels (original).", { nullable: true, placeholder: `all (${value.n_channels})`, min: 1 })}
          {num("pool1_kernel", "Pool 1 size", undefined, { min: 1 })}
          {num("pool1_stride", "Pool 1 stride", "Empty = pool size.", { nullable: true, placeholder: "= size", min: 1 })}
          {num("separable_kernel", "Separable kernel", undefined, { min: 1 })}
          {num("pool2_kernel", "Pool 2 size", undefined, { min: 1 })}
          {num("pool2_stride", "Pool 2 stride", "Empty = pool size.", { nullable: true, placeholder: "= size", min: 1 })}
          <SelectField label="Pooling" value={value.pool_type} disabled={disabled} changed={ch("pool_type")} onChange={(v) => set({ pool_type: v })} options={[{ value: "avg", label: "Average" }, { value: "max", label: "Max" }]} />
          <SelectField label="Padding" value={value.padding} disabled={disabled} changed={ch("padding")} onChange={(v) => set({ padding: v })} options={[{ value: "same", label: "same" }, { value: "valid", label: "valid" }]} />
          <SelectField label="Activation" value={value.activation} disabled={disabled} changed={ch("activation")} onChange={(v) => set({ activation: v })} options={[{ value: "elu", label: "ELU" }, { value: "relu", label: "ReLU" }, { value: "leaky_relu", label: "Leaky ReLU" }, { value: "gelu", label: "GELU" }, { value: "silu", label: "SiLU" }]} />
          {num("bn_eps", "BatchNorm ε", undefined, { step: 0.0001 })}
        </div>
        <div className="mt-3">
          <Toggle label="Batch normalisation" checked={value.batch_norm} disabled={disabled} onChange={(v) => set({ batch_norm: v })} />
        </div>
      </fieldset>
      <fieldset>
        <legend className="mb-1 text-[12px] font-bold uppercase tracking-[0.07em] text-ink-subtle">Training-time regularisation</legend>
        <p className="mb-3 text-[12.5px] text-ink-subtle">Active only while training; they do not change weight shapes or inference outputs.</p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {num("dropout", "Dropout p", "0.5 within-subject, 0.25 cross-subject in the original paper.", { step: 0.05, min: 0 })}
          <SelectField label="Dropout type" value={value.dropout_type} disabled={disabled} changed={ch("dropout_type")} onChange={(v) => set({ dropout_type: v })} options={[{ value: "dropout", label: "Element-wise" }, { value: "spatial", label: "Spatial (whole maps)" }]} />
          {num("bn_momentum", "BN momentum", "PyTorch convention (0.01 ≈ Keras 0.99).", { step: 0.01 })}
          {num("depthwise_max_norm", "Depthwise max-norm", "Empty disables the constraint.", { nullable: true, step: 0.05, placeholder: "off" })}
          {num("classifier_max_norm", "Classifier max-norm", "Empty disables the constraint.", { nullable: true, step: 0.05, placeholder: "off" })}
        </div>
      </fieldset>
    </div>
  );
}
