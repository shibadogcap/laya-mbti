import { For } from "solid-js";
import { fromDateInput, toDateInput } from "../lib/format.js";

export interface PeriodPickerProps {
  minTs: number;
  maxTs: number;
  from: number | null;
  to: number | null;
  setFrom: (value: number | null) => void;
  setTo: (value: number | null) => void;
  count: number;
  total: number;
}

interface Preset {
  label: string;
  from: (max: number) => number | null;
}

const DAY = 24 * 60 * 60 * 1000;

const PRESETS: Preset[] = [
  { label: "全期間", from: () => null },
  { label: "直近1年", from: (max) => max - 365 * DAY },
  { label: "直近3か月", from: (max) => max - 90 * DAY },
  { label: "直近1か月", from: (max) => max - 30 * DAY },
];

export function PeriodPicker(props: PeriodPickerProps) {
  const activePreset = () => {
    if (props.to !== null) return "";
    if (props.from === null) return "全期間";
    const presets: Record<string, number> = {
      "直近1年": props.maxTs - 365 * DAY,
      "直近3か月": props.maxTs - 90 * DAY,
      "直近1か月": props.maxTs - 30 * DAY,
    };
    return Object.entries(presets).find(([, value]) => value === props.from)?.[0] ?? "";
  };

  return (
    <section class="setup-panel setup-panel--period setup-panel--compact">
      <header class="panel__header setup-panel__header">
        <h2>対象期間</h2>
        <strong class="panel__meta setup-panel__count numeric">
          {props.count.toLocaleString()} / {props.total.toLocaleString()} 件
        </strong>
      </header>

      <div class="period period--compact">
        <div class="period__presets period__presets--compact">
          <For each={PRESETS}>
            {(preset) => (
              <button
                type="button"
                class="chip chip--preset"
                classList={{ "chip--active": activePreset() === preset.label }}
                aria-pressed={activePreset() === preset.label}
                onClick={() => {
                  props.setFrom(preset.from(props.maxTs));
                  props.setTo(null);
                }}
              >
                {preset.label}
              </button>
            )}
          </For>
        </div>

        <div class="period__inputs period__inputs--compact">
          <label class="period__field">
            <span>開始</span>
            <input
              type="date"
              min={toDateInput(props.minTs)}
              max={toDateInput(props.maxTs)}
              value={toDateInput(props.from)}
              onInput={(event) =>
                props.setFrom(fromDateInput(event.currentTarget.value, false))
              }
            />
          </label>
          <label class="period__field">
            <span>終了</span>
            <input
              type="date"
              min={toDateInput(props.minTs)}
              max={toDateInput(props.maxTs)}
              value={toDateInput(props.to)}
              onInput={(event) =>
                props.setTo(fromDateInput(event.currentTarget.value, true))
              }
            />
          </label>
          <button
            type="button"
            class="chip chip--ghost period__clear"
            aria-label="期間をクリア"
            title="期間をクリア"
            onClick={() => {
              props.setFrom(null);
              props.setTo(null);
            }}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M20 11a8 8 0 1 0 1 4" />
              <path d="M20 4v7h-7" />
            </svg>
          </button>
        </div>
      </div>
    </section>
  );
}
