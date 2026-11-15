/*
 * Inference panel: live pipeline diagnostics from GET /api/diagnostics.
 *
 * This panel replaced the former mock pose/depth/flow/segmentation viewer.
 * It shows the reranker decision, the reuse/refresh counters, stage timings,
 * splat and mask summary numbers, and the stream/queue health.
 */

import type { DiagnosticsPayload } from '../api/types'

interface RowSpec {
  key: string
  label: string
}

interface SectionSpec {
  title: string
  rows: RowSpec[]
}

const SECTIONS: SectionSpec[] = [
  {
    title: 'Decision',
    rows: [
      { key: 'action', label: 'Action' },
      { key: 'reason', label: 'Reason' },
      { key: 'flow', label: 'Flow / threshold' },
      { key: 'cache', label: 'Cache age' },
      { key: 'frames', label: 'Selected / active' },
      { key: 'candidates', label: 'Candidates' },
    ],
  },
  {
    title: 'Refresh chain',
    rows: [
      { key: 'refresh', label: 'Refresh ok / failed' },
      { key: 'timings', label: 'Timings' },
      { key: 'worker', label: 'Worker' },
      { key: 'error', label: 'Last error' },
    ],
  },
  {
    title: 'Splat state',
    rows: [
      { key: 'splat', label: 'Splat frame / count' },
      { key: 'mask', label: 'Mask area / conf.' },
      { key: 'maskRange', label: 'Mask near - far' },
      { key: 'pose', label: 'Pose / person' },
      { key: 'depthSource', label: 'Depth source' },
    ],
  },
  {
    title: 'Stream',
    rows: [
      { key: 'stream', label: 'Stream health' },
      { key: 'queue', label: 'Queue depth / drops' },
      { key: 'counters', label: 'Counters' },
    ],
  },
]

export class InferencePanel {
  private element: HTMLElement
  private values = new Map<string, HTMLElement>()

  constructor() {
    this.element = document.createElement('section')
    this.element.className = 'viewer-panel'

    const sections = SECTIONS.map(
      (section) => `
        <div class="diag-section">
          <h3>${section.title}</h3>
          ${section.rows
            .map(
              (row) => `
                <div class="diag-row">
                  <span class="diag-label">${row.label}</span>
                  <span class="diag-value" data-key="${row.key}">--</span>
                </div>
              `,
            )
            .join('')}
        </div>
      `,
    ).join('')

    this.element.innerHTML = `
      <div class="panel-header">
        <h2>Inference Diagnostics</h2>
        <span class="panel-note" id="diag-health">waiting</span>
      </div>

      <div class="viewer-content diagnostics-content">
        <div class="diagnostics">${sections}</div>
      </div>

      <div class="panel-footer">
        <span class="inference-footer-item">Live from /api/diagnostics</span>
        <span class="inference-footer-item">Poll: 300 ms</span>
      </div>
    `

    this.element
      .querySelectorAll<HTMLElement>('.diag-value')
      .forEach((value) => {
        this.values.set(value.dataset.key ?? '', value)
      })
  }

  getElement(): HTMLElement {
    return this.element
  }

  update(diagnostics: DiagnosticsPayload | null): void {
    if (diagnostics === null) {
      this.values.forEach((value) => {
        value.textContent = '--'
        value.classList.remove('is-error', 'is-good', 'is-warn')
      })
      this.setHealth('waiting')
      return
    }

    const splat = diagnostics.splat
    const mask = splat?.mask ?? null
    const timings = splat?.timings_ms ?? null

    this.set('action', formatAction(diagnostics.action))
    this.set('reason', diagnostics.reason ?? '--')
    this.set(
      'flow',
      `${formatNumber(diagnostics.flow_score, 4)} / ${formatNumber(
        diagnostics.flow_threshold,
        3,
      )}`,
    )
    this.set(
      'cache',
      diagnostics.cache_age_ms === null
        ? '--'
        : `${diagnostics.cache_age_ms.toFixed(1)} ms`,
    )
    this.set(
      'frames',
      `selected ${diagnostics.selected_frame_id ?? '--'}  ·  active ${
        diagnostics.active_state_frame_id ?? '--'
      }`,
    )
    this.set('candidates', diagnostics.candidates?.toString() ?? '--')

    this.set(
      'refresh',
      `${diagnostics.refresh_count} ok  ·  ${diagnostics.refresh_failures} failed`,
    )
    this.set(
      'timings',
      timings === null
        ? '--'
        : `pose+depth ${formatNumber(timings.pose_depth, 1)} ms  ·  mask ${formatNumber(
            timings.mask,
            1,
          )} ms  ·  construct ${formatNumber(timings.construct, 1)} ms  ·  total ${formatNumber(
            timings.total,
            1,
          )} ms`,
    )
    this.set(
      'worker',
      diagnostics.refresh_worker === null
        ? '--'
        : `active ${diagnostics.refresh_worker.active ? 'yes' : 'no'}  ·  submitted ${
            diagnostics.refresh_worker.submitted_total ?? 0
          }  ·  completed ${diagnostics.refresh_worker.completed_total ?? 0}  ·  replaced ${
            diagnostics.refresh_worker.replaced_total ?? 0
          }`,
    )
    this.set(
      'error',
      diagnostics.last_error ?? 'none',
      diagnostics.last_error === null ? 'is-good' : 'is-error',
    )

    this.set(
      'splat',
      splat === null ? '--' : `frame ${splat.frame_id ?? '--'}  ·  ${splat.count ?? '--'} splats`,
    )
    this.set(
      'mask',
      mask === null
        ? '--'
        : `area ${formatPercent(mask.area_fraction)}  ·  conf ${formatNumber(
            mask.confidence,
            2,
          )}`,
    )
    this.set(
      'maskRange',
      mask === null || mask.near_m === null || mask.far_m === null
        ? '--'
        : `${mask.near_m.toFixed(2)} m - ${mask.far_m.toFixed(2)} m`,
    )
    this.set(
      'pose',
      splat === null
        ? '--'
        : `confidence ${formatNumber(splat.pose_confidence, 2)}  ·  person ${
            splat.person_detected ? 'yes' : 'no'
          }`,
      splat?.person_detected ? 'is-good' : 'is-warn',
    )
    this.set('depthSource', splat?.depth_source ?? '--')

    this.set('stream', String(diagnostics.stream_health))
    this.set(
      'queue',
      `depth ${diagnostics.queue.depth}  ·  drops ${diagnostics.queue.dropped_total}  ·  skipped ${diagnostics.queue.skipped_total}`,
    )
    const counters = diagnostics.counters
    this.set(
      'counters',
      `frames ${counters.frames ?? 0}  ·  windows ${counters.windows ?? 0}  ·  refresh ${
        counters.refresh_requested ?? 0
      }  ·  reuse ${counters.reuse ?? 0}`,
    )

    this.setHealth(diagnostics.stream_health)
  }

  private set(
    key: string,
    text: string,
    tone: 'is-error' | 'is-good' | 'is-warn' | null = null,
  ): void {
    const value = this.values.get(key)
    if (!value) {
      return
    }
    value.textContent = text
    value.classList.remove('is-error', 'is-good', 'is-warn')
    if (tone !== null) {
      value.classList.add(tone)
    }
  }

  private setHealth(health: string): void {
    const element = this.element.querySelector<HTMLElement>('#diag-health')
    if (!element) {
      return
    }
    element.textContent = health
    element.classList.toggle('is-live', health === 'healthy')
    element.classList.toggle('is-error', health === 'closed')
  }
}

function formatAction(action: string | null): string {
  if (action === null) {
    return '--'
  }
  return action.toUpperCase()
}

function formatNumber(
  value: number | null | undefined,
  digits: number,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '--'
  }
  return value.toFixed(digits)
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '--'
  }
  return `${(value * 100).toFixed(1)}%`
}
