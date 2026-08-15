import axiosClient from "@/api/axiosClient";
import { useCallback, useRef } from "react";

import {
  IdempotencyRegistry,
  shouldReleaseIdempotencyKey,
} from "./idempotencyRegistry";

export function useApi() {
  const idempotencyRegistry = useRef(new IdempotencyRegistry());

  const get = useCallback(async (url: string, config = {}) => {
    const res = await axiosClient.get(url, config);
    return res.data;
  }, []);

  const post = useCallback(
    async (url: string, data?: any, config?: { isBlob?: boolean }) => {
      const res = await axiosClient.post(url, data, {
        responseType: config?.isBlob ? "blob" : "json",
      });

      if (config?.isBlob) return res;
      return res.data;
    },
    [],
  );

  const postIdempotent = useCallback(async (url: string, data?: unknown) => {
    const fingerprint = JSON.stringify([url, data]);
    const key = idempotencyRegistry.current.getOrCreate(fingerprint);

    const response = await axiosClient.post(url, data, {
      headers: { "Idempotency-Key": key },
    });

    // Trạng thái cần đối soát vẫn phải giữ key để không tạo mutation mới.
    if (shouldReleaseIdempotencyKey(response.data)) {
      idempotencyRegistry.current.release(fingerprint, key);
    }
    return response.data;
  }, []);

  const put = useCallback(async (url: string, data: any) => {
    const res = await axiosClient.put(url, data);
    return res.data;
  }, []);

  const remove = useCallback(async (url: string) => {
    const res = await axiosClient.delete(url);
    return res.data;
  }, []);

  return { get, post, postIdempotent, put, remove };
}
