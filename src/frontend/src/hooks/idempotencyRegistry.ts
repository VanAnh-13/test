type KeyFactory = () => string;

export class IdempotencyRegistry {
  private readonly keys = new Map<string, string>();
  private readonly createKey: KeyFactory;

  constructor(createKey: KeyFactory = () => crypto.randomUUID()) {
    this.createKey = createKey;
  }

  getOrCreate(fingerprint: string): string {
    const existingKey = this.keys.get(fingerprint);
    if (existingKey) return existingKey;

    const key = this.createKey();
    this.keys.set(fingerprint, key);
    return key;
  }

  release(fingerprint: string, expectedKey: string): void {
    if (this.keys.get(fingerprint) === expectedKey) {
      this.keys.delete(fingerprint);
    }
  }
}

export function shouldReleaseIdempotencyKey(payload: unknown): boolean {
  if (!payload || typeof payload !== "object") return false;
  return Reflect.get(payload, "status") === "success";
}
