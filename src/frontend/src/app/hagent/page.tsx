import type { Metadata } from "next";

import { HAgentWorkspace } from "@/components/hagentWorkspace/HAgentWorkspace";

export const metadata: Metadata = {
  title: "HAgent | HAutoML",
  description: "Không gian điều hành quy trình AutoML có kiểm soát.",
};

export default function HAgentPage() {
  return <HAgentWorkspace />;
}
