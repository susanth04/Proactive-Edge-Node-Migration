"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  ["/", "Dashboard"],
  ["/simulate", "Simulate"],
  ["/forecast", "Forecast"],
  ["/sensitivity", "Sensitivity"],
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <>
      <header className="topbar">
        <Link className="wordmark" href="/">EDGE NETWORK SIMULATOR</Link>
        <nav aria-label="Main navigation">
          {links.map(([href, label]) => (
            <Link key={href} href={href} aria-current={pathname === href ? "page" : undefined}>{label}</Link>
          ))}
        </nav>
        <span className="edition">LEARNING TOOL · GENERATED DATA</span>
      </header>
      <main className="page" key={pathname}>{children}</main>
      <footer className="footer">
        <span>Explore how nearby computers share workloads.</span>
        <span>SIMULATED DATA · NOT LIVE MONITORING</span>
      </footer>
    </>
  );
}

export function PageHeading({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) {
  return (
    <div className="page-heading">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p className="lead">{detail}</p>
    </div>
  );
}

export function ErrorMessage({ message }: { message: string }) {
  return <p className="error-message" role="alert">{message}</p>;
}

export function LoadingMessage() {
  return <p className="status-message" role="status">Reading simulator results…</p>;
}

export function DataTable({ rows, columns }: { rows: Record<string, unknown>[]; columns?: string[] }) {
  if (!rows.length) return <p className="status-message">No rows are available for this selection.</p>;
  const keys = columns ?? Object.keys(rows[0]);
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead><tr>{keys.map((key) => <th key={key} scope="col">{key}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>{keys.map((key) => (
              <td className={typeof row[key] === "number" ? "num" : ""} key={key}>
                {typeof row[key] === "number" ? Number(row[key]).toLocaleString(undefined, { maximumSignificantDigits: 5 }) : String(row[key] ?? "—")}
              </td>
            ))}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
