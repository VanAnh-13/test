import {
  Activity,
  BarChart3,
  Database,
  FlaskConical,
  Plus,
  Rocket,
  Sparkles,
} from "lucide-react";

import { Button } from "@/components/ui/button";

const journeyStages = [
  { label: "Kiểm tra dữ liệu", icon: Database },
  { label: "Thiết kế thử nghiệm", icon: FlaskConical },
  { label: "Huấn luyện", icon: Activity },
  { label: "Đánh giá", icon: BarChart3 },
  { label: "Ứng viên phát hành", icon: Rocket },
] as const;

interface ConversationRailProps {
  runId: string | null;
  statusLabel: string;
  lastSequence: number;
  onNewRun: () => void;
}

export function ConversationRail({
  runId,
  statusLabel,
  lastSequence,
  onNewRun,
}: ConversationRailProps) {
  return (
    <aside
      aria-label="Phiên làm việc HAgent"
      className="border-b border-border bg-muted/30 p-4 lg:border-b-0 lg:border-r"
    >
      <div className="flex items-center justify-between gap-3 lg:block">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            HAgent workspace
          </p>
          <h2 className="mt-1 text-sm font-semibold">Phiên AutoML</h2>
        </div>
        <Button
          type="button"
          size="sm"
          className="lg:mt-4 lg:w-full"
          onClick={onNewRun}
        >
          <Plus aria-hidden="true" />
          Phiên mới
        </Button>
      </div>

      <section className="mt-5" aria-labelledby="current-run-heading">
        <h3
          id="current-run-heading"
          className="text-xs font-medium uppercase tracking-wide text-muted-foreground"
        >
          Đang theo dõi
        </h3>
        {runId ? (
          <div className="mt-2 border-l-2 border-amber-500 bg-background px-3 py-3">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Sparkles className="size-4 text-amber-600" aria-hidden="true" />
              {statusLabel}
            </div>
            <p className="mt-2 truncate font-mono text-[11px] text-muted-foreground">
              {runId}
            </p>
            <p className="mt-1 font-mono text-[11px] text-muted-foreground">
              sequence {lastSequence}
            </p>
          </div>
        ) : (
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Chưa có run. Hãy mô tả mục tiêu ở vùng hội thoại.
          </p>
        )}
      </section>

      <section className="mt-6 hidden lg:block" aria-labelledby="journey-heading">
        <h3
          id="journey-heading"
          className="text-xs font-medium uppercase tracking-wide text-muted-foreground"
        >
          Hành trình chuẩn
        </h3>
        <ol className="mt-3 space-y-1">
          {journeyStages.map(({ label, icon: Icon }, index) => (
            <li key={label} className="flex items-center gap-3 py-2 text-xs">
              <span className="font-mono text-muted-foreground">
                {String(index + 1).padStart(2, "0")}
              </span>
              <Icon className="size-4 text-muted-foreground" aria-hidden="true" />
              <span>{label}</span>
            </li>
          ))}
        </ol>
      </section>
    </aside>
  );
}
