export interface ServerAuthEnvironment {
  readonly AUTH_API_BASE_URL?: string;
  readonly SESSION_HTTPS_ONLY?: string;
  readonly NODE_ENV?: string;
}

type SearchParameters = Readonly<Record<string, string>>;

function parseAuthApiBaseUrl(environment: ServerAuthEnvironment): URL {
  const configured = environment.AUTH_API_BASE_URL?.trim();
  if (!configured) {
    throw new Error("AUTH_API_BASE_URL chưa được cấu hình");
  }

  let url: URL;
  try {
    url = new URL(configured);
  } catch {
    throw new Error("AUTH_API_BASE_URL phải là URL tuyệt đối");
  }

  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("AUTH_API_BASE_URL chỉ hỗ trợ HTTP hoặc HTTPS");
  }
  if (url.username || url.password) {
    throw new Error("AUTH_API_BASE_URL không được chứa credential");
  }
  if (url.search || url.hash) {
    throw new Error("AUTH_API_BASE_URL không được chứa query hoặc fragment");
  }
  if (url.pathname !== "/") {
    throw new Error("AUTH_API_BASE_URL không được chứa base path");
  }
  return url;
}

export function resolveAuthApiUrl(
  pathname: string,
  searchParameters?: SearchParameters,
  environment: ServerAuthEnvironment = process.env,
): URL {
  if (!pathname.startsWith("/") || pathname.startsWith("//")) {
    throw new Error("Auth API pathname không hợp lệ");
  }

  const url = new URL(pathname, parseAuthApiBaseUrl(environment));
  for (const [name, value] of Object.entries(searchParameters ?? {})) {
    url.searchParams.set(name, value);
  }
  return url;
}

export function resolvePasswordResetCookieSecure(
  environment: ServerAuthEnvironment = process.env,
): boolean {
  const configured = environment.SESSION_HTTPS_ONLY?.trim().toLowerCase();
  if (configured === "true") return true;
  if (configured === "false") return false;
  if (configured !== undefined || environment.NODE_ENV === "production") {
    throw new Error("SESSION_HTTPS_ONLY phải được cấu hình bằng true hoặc false");
  }
  return false;
}
