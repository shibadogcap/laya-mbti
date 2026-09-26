import type { RankedType } from "./types.js";

/**
 * The one number the whole result is read through: the mean of the four axis
 * probabilities of a type. The posterior product stays the sort key, because it
 * is what makes the candidate ordering a distribution, but a product of four
 * factors is unreadable next to a mean and made the headline look far weaker
 * than the evidence.
 */
export function typeAgreement(entry: RankedType | undefined): number {
  if (!entry) return 0;
  const values = entry.axes
    .map((fit) => fit.matchProbability)
    .filter((value) => Number.isFinite(value));
  if (values.length === 0) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

/** Renders a probability so that 1% and "under 1%" stay distinguishable. */
export function formatPercent(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0%";
  const percent = value * 100;
  if (percent < 1) return "<1%";
  return `${Math.round(percent)}%`;
}
