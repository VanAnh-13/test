import { expect, test, type Page } from "@playwright/test";

function observePublicFailures(page: Page) {
  const failures: string[] = [];

  page.on("pageerror", (error) => {
    failures.push(`page:${error.name}`);
  });
  page.on("console", (message) => {
    if (message.type() === "error") failures.push("console:error");
  });
  page.on("response", (response) => {
    if (response.status() >= 500) {
      failures.push(`http:${response.status()}:${new URL(response.url()).pathname}`);
    }
  });

  return failures;
}

async function openWorkspace(page: Page) {
  await page.goto("/hagent", { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle");
}

test("hiển thị đầy đủ workspace HAgent khi chưa đăng nhập", async ({ page }) => {
  const failures = observePublicFailures(page);

  await openWorkspace(page);

  await expect(page).toHaveTitle("HAgent | HAutoML");
  await expect(page.getByLabel("Phiên làm việc HAgent")).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "Từ mục tiêu đến mô hình, có bằng chứng ở từng bước.",
    }),
  ).toBeVisible();
  await expect(page.getByLabel("Run ledger")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Đăng nhập", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Chưa có run. Hãy mô tả mục tiêu ở vùng hội thoại.")).toBeVisible();
  expect(failures).toEqual([]);
});

test("giữ ba vùng workspace trong viewport mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const failures = observePublicFailures(page);

  await openWorkspace(page);

  await expect(page.getByLabel("Phiên làm việc HAgent")).toBeVisible();
  await expect(page.getByRole("main")).toBeVisible();
  await expect(page.getByLabel("Run ledger")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Đăng nhập", exact: true }),
  ).toBeVisible();
  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(hasHorizontalOverflow).toBe(false);
  expect(failures).toEqual([]);
});
