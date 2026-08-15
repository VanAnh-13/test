"use client";

import { FormEvent, KeyboardEvent, useState } from "react";
import { ArrowUp, LoaderCircle, LogIn, MessageSquareText } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

export interface WorkspaceMessage {
  readonly id: string;
  readonly role: "user" | "assistant";
  readonly content: string;
}

interface ChatCanvasProps {
  messages: readonly WorkspaceMessage[];
  isAuthenticated: boolean;
  isBusy: boolean;
  statusLabel: string;
  error: string | null;
  onSubmit: (message: string) => Promise<void>;
  onSignIn: () => void;
}

const promptExamples = [
  "Kiểm tra chất lượng dataset và xác định target phù hợp",
  "Thiết kế thử nghiệm phân loại với ngân sách nhỏ",
] as const;

export function ChatCanvas({
  messages,
  isAuthenticated,
  isBusy,
  statusLabel,
  error,
  onSubmit,
  onSignIn,
}: ChatCanvasProps) {
  const [draft, setDraft] = useState("");

  const send = async () => {
    const message = draft.trim();
    if (!message || isBusy || !isAuthenticated) return;
    setDraft("");
    await onSubmit(message);
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void send();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  return (
    <main className="flex min-h-[32rem] min-w-0 flex-col bg-background">
      <header className="border-b border-border px-5 py-4 sm:px-8">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-amber-700 dark:text-amber-400">
          AutoML control room
        </p>
        <h1 className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">
          Từ mục tiêu đến mô hình, có bằng chứng ở từng bước.
        </h1>
      </header>

      <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8" aria-live="polite">
        {messages.length === 0 ? (
          <section className="mx-auto flex h-full max-w-2xl flex-col justify-center">
            <MessageSquareText
              className="size-8 text-amber-600"
              aria-hidden="true"
            />
            <h2 className="mt-4 text-lg font-semibold">Bạn muốn HAgent xử lý gì?</h2>
            <p className="mt-2 max-w-xl text-sm leading-6 text-muted-foreground">
              Mô tả dataset, target và kết quả mong muốn. HAgent sẽ kiểm tra dữ
              liệu, đề xuất ExperimentSpec và luôn chờ bạn duyệt trước khi train.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              {promptExamples.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="rounded-md border border-border bg-muted/40 px-3 py-2 text-left text-xs transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none"
                  onClick={() => setDraft(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </section>
        ) : (
          <ol className="mx-auto max-w-3xl space-y-6">
            {messages.map((message) => (
              <li
                key={message.id}
                className={message.role === "user" ? "ml-auto max-w-2xl" : "mr-auto max-w-3xl"}
              >
                <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  {message.role === "user" ? "Bạn" : "HAgent"}
                </p>
                <div
                  className={
                    message.role === "user"
                      ? "rounded-md bg-foreground px-4 py-3 text-sm leading-6 text-background"
                      : "border-l-2 border-amber-500 px-4 py-1 text-sm leading-7"
                  }
                >
                  {message.content}
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>

      <div className="border-t border-border bg-background px-4 py-4 sm:px-8">
        {error && (
          <p role="alert" className="mx-auto mb-3 max-w-3xl text-sm text-destructive">
            {error}
          </p>
        )}
        {!isAuthenticated ? (
          <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 border border-border bg-muted/30 p-4">
            <p className="text-sm">Đăng nhập để bắt đầu một run có owner scope.</p>
            <Button type="button" size="sm" onClick={onSignIn}>
              <LogIn aria-hidden="true" />
              Đăng nhập
            </Button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
            <label htmlFor="hagent-message" className="sr-only">
              Mục tiêu AutoML
            </label>
            <div className="flex items-end gap-2 rounded-lg border border-input bg-muted/20 p-2 focus-within:ring-1 focus-within:ring-ring">
              <Textarea
                id="hagent-message"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ví dụ: audit dataset churn và đề xuất metric phù hợp..."
                className="max-h-40 min-h-12 resize-y border-0 bg-transparent shadow-none focus-visible:ring-0"
                disabled={isBusy}
                maxLength={32768}
              />
              <Button
                type="submit"
                size="icon"
                aria-label="Gửi mục tiêu"
                disabled={!draft.trim() || isBusy}
              >
                {isBusy ? (
                  <LoaderCircle className="animate-spin motion-reduce:animate-none" aria-hidden="true" />
                ) : (
                  <ArrowUp aria-hidden="true" />
                )}
              </Button>
            </div>
            <p className="mt-2 text-center text-[11px] text-muted-foreground" aria-live="polite">
              {statusLabel} · Enter để gửi, Shift + Enter để xuống dòng
            </p>
          </form>
        )}
      </div>
    </main>
  );
}
