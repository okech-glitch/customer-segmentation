import axios, { AxiosError } from "axios";

const DEFAULT_LOCAL_BACKEND = "http://localhost:8000";

export function getApiBaseUrl(): string {
  const envUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL?.trim() ||
    process.env.API_BASE_URL?.trim();

  if (envUrl) {
    return envUrl.replace(/\/$/, "");
  }

  if (typeof window === "undefined") {
    // During SSR or build, fall back to localhost unless an env override exists
    return DEFAULT_LOCAL_BACKEND;
  }

  const protocol = window.location.protocol;
  const hostname = window.location.hostname;
  const port = window.location.port;

  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return DEFAULT_LOCAL_BACKEND;
  }

  const portSuffix = port ? `:${port}` : "";
  return `${protocol}//${hostname}${portSuffix}`;
}

export const api = axios.create({
  baseURL: getApiBaseUrl(),
  headers: {
    "Content-Type": "application/json",
  },
});

export function normalizeAxiosError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const err = error as AxiosError<{ detail?: unknown; message?: string } | string>;
    const data = err.response?.data;

    if (typeof data === "string") {
      return data;
    }

    const detail =
      typeof data === "object" && data !== null
        ? (data as { detail?: unknown; message?: string }).detail ??
          (data as { detail?: unknown; message?: string }).message
        : undefined;

    if (Array.isArray(detail)) {
      return detail.join("\n");
    }

    if (typeof detail === "string") {
      return detail;
    }

    return err.message || "Request failed";
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "An unexpected error occurred";
}
