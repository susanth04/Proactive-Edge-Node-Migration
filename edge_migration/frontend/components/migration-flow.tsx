"use client";

import { useEffect, useState } from "react";
import type { Policy, PolicyTimeline, SimulationResponse } from "@/lib/api";
import { useReducedMotion } from "@/lib/use-reduced-motion";

function nearestPoint(data: PolicyTimeline, slot: number) {
  return data.points.reduce((nearest, point) =>
    Math.abs(point.slot - slot) < Math.abs(nearest.slot - slot) ? point : nearest,
  data.points[0]);
}

export function MigrationFlow({
  result, policy, selectedNode, onSelectNode,
}: {
  result: SimulationResponse;
  policy: Policy;
  selectedNode: number;
  onSelectNode: (node: number) => void;
}) {
  const data = result.timeline.policies[policy];
  const events = data?.migrations ?? [];
  const [eventIndex, setEventIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    setEventIndex(0);
    setPlaying(false);
  }, [policy, result]);
  useEffect(() => {
    if (!playing || events.length < 2) return;
    const timer = window.setInterval(() => setEventIndex((index) => (index + 1) % events.length), 1900);
    return () => window.clearInterval(timer);
  }, [events.length, playing]);
  if (!data || !data.points.length) return <p className="status-message">No topology timeline is available.</p>;

  const active = events[eventIndex];
  const nodeCount = result.timeline.node_count;
  const columns = Math.min(4, nodeCount);
  const rows = Math.ceil(nodeCount / columns);
  const width = Math.max(580, columns * 155 + 55);
  const height = rows * 150 + 42;
  const positions = Array.from({ length: nodeCount }, (_, id) => ({
    x: 45 + (id % columns) * 155,
    y: 47 + Math.floor(id / columns) * 150,
  }));
  const loadPoint = active ? nearestPoint(data, active.slot) : data.points[0];
  const source = active ? positions[active.source] : undefined;
  const destination = active ? positions[active.destination] : undefined;
  const path = source && destination ? `M ${source.x} ${source.y} L ${destination.x} ${destination.y}` : "";
  const selectedLoad = Number(loadPoint[`node_${selectedNode}`] ?? 0);
  const movesFrom = events.filter((event) => event.source === selectedNode);
  const movesTo = events.filter((event) => event.destination === selectedNode);
  const nodeEvents = events.filter((event) => event.source === selectedNode || event.destination === selectedNode);

  return (
    <div>
      <div className="flow-controls">
        <button className="button button-primary" type="button" disabled={!events.length}
          onClick={() => setPlaying((value) => !value)} aria-pressed={playing}>
          {playing ? "Pause animation" : "Play moves"}
        </button>
        <button className="button" type="button" disabled={!events.length}
          onClick={() => setEventIndex((index) => events.length ? (index + 1) % events.length : 0)}>
          Show next move
        </button>
        <span className="flow-caption">
          {active
            ? `Move ${eventIndex + 1} of ${events.length}: workload ${active.service + 1} moves from computer ${active.source + 1} to computer ${active.destination + 1}.`
            : "This rule did not need to move any apps in this run."}
        </span>
      </div>
      <div className="flow-layout">
        <div className="flow-canvas">
        <svg viewBox={`0 0 ${width} ${height}`} role="group" aria-label="Click a computer to inspect its workload and app moves">
          {positions.map((point, index) => {
            const next = positions[(index + 1) % nodeCount];
            return index < nodeCount - 1
              ? <line className="flow-link" key={`link-${index}`} x1={point.x} y1={point.y} x2={next.x} y2={next.y} />
              : null;
          })}
          {active && <path className="flow-transfer" d={path} />}
          {active && source && destination && (
            <g key={`${policy}-${eventIndex}`}>
              {reducedMotion
                ? <circle className="flow-packet" cx={destination.x} cy={destination.y} r="9" />
                : <circle className="flow-packet" r="9">
                    <animateMotion dur="1.6s" fill="freeze" path={path} />
                  </circle>}
            </g>
          )}
          {positions.map((point, node) => {
            const value = Number(loadPoint[`node_${node}`] ?? 0);
            const role = active?.source === node ? "source" : active?.destination === node ? "destination" : "";
            const selected = selectedNode === node;
            return (
              <g key={node} className="flow-node-hit" role="button" tabIndex={0} aria-pressed={selected}
                aria-label={`Computer ${node + 1}, ${Math.round(value * 100)} percent busy${selected ? ", selected" : ""}. Select for details.`}
                onClick={() => onSelectNode(node)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelectNode(node);
                  }
                }}>
                <circle className={`flow-node ${role ? `flow-node--${role}` : ""} ${selected ? "flow-node--selected" : ""}`}
                  cx={point.x} cy={point.y} r="35" />
                <text className="flow-node-number" x={point.x} y={point.y - 3}>Computer {node + 1}</text>
                <text className="flow-node-load" x={point.x} y={point.y + 15}>{(value * 100).toFixed(0)}% busy</text>
              </g>
            );
          })}
        </svg>
        </div>
        <aside className="flow-detail" aria-live="polite">
          <div className="flow-detail-heading">
            <h3>Computer {selectedNode + 1}</h3>
            <span>At minute {loadPoint.slot}</span>
          </div>
          <dl className="flow-metrics">
            <div><dt>Computer load</dt><dd>{(selectedLoad * 100).toFixed(1)}%</dd></div>
            <div><dt>Apps moved away</dt><dd>{movesFrom.length}</dd></div>
            <div><dt>Apps moved here</dt><dd>{movesTo.length}</dd></div>
          </dl>
          <h4>App moves involving this computer</h4>
          {nodeEvents.length ? (
            <ol className="flow-event-list">
              {nodeEvents.map((event, index) => (
                <li key={`${event.slot}-${event.service}-${index}`}>
                  Minute {event.slot}: workload {event.service + 1} moves from computer {event.source + 1} to computer {event.destination + 1}.
                </li>
              ))}
            </ol>
          ) : <p className="chart-caption">No apps moved to or from this computer in this run.</p>}
        </aside>
      </div>
      <p className="chart-caption">Percentages show the computer's combined workload at the selected move. The animated transfers come from the measured run; the faint connection lines are illustrative.</p>
    </div>
  );
}
