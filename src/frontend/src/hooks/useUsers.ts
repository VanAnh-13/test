// Cung cấp danh sách người dùng và thao tác tải lại theo yêu cầu.

import { useCallback, useEffect, useState } from "react";
import { useApi } from "./useApi";

export type User = {
  _id: string;
  username: string;
  email: string;
  password: string;
  gender: string;
  date: string;
  number: string;
  role: string;
  fullName: string;
};

export default function useUsers() {
  const { get } = useApi();
  const [users, setUsers] = useState<User[]>([]);

  const fetchUsers = useCallback(async () => {
    try {
      const data = await get(`${process.env.NEXT_PUBLIC_BASE_API}/users`);
      setUsers(data);
    } catch (error) {
      console.error("Không thể tải danh sách người dùng:", error);
    }
  }, [get]);

  useEffect(() => {
    void fetchUsers();
  }, [fetchUsers]);

  return { users, fetchUsers };
}
