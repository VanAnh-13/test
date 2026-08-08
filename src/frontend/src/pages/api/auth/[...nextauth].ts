import NextAuth, { NextAuthOptions, User } from "next-auth";
import { JWT } from "next-auth/jwt";
import CredentialsProvider from "next-auth/providers/credentials";
import { jwtDecode } from "jwt-decode";

const AUTH_API_BASE_URL =
  process.env.AUTH_API_BASE_URL || process.env.NEXT_PUBLIC_BASE_API;

type TokenResponse = {
  access_token: string;
  refresh_token: string;
};

type ApiUser = {
  _id: string;
  username: string;
  email: string;
  role: string;
};

function parseTokenResponse(value: unknown): TokenResponse {
  if (
    typeof value !== "object" ||
    value === null ||
    !("access_token" in value) ||
    typeof value.access_token !== "string" ||
    !("refresh_token" in value) ||
    typeof value.refresh_token !== "string"
  ) {
    throw new Error("Phản hồi token không hợp lệ");
  }

  return {
    access_token: value.access_token,
    refresh_token: value.refresh_token,
  };
}

function getAccessTokenExpiry(accessToken: string): number {
  const decoded = jwtDecode<{ exp?: number }>(accessToken);
  if (typeof decoded.exp !== "number") {
    throw new Error("Access token không có thời hạn hợp lệ");
  }
  return decoded.exp * 1000;
}

async function requestTokens(
  path: string,
  body: Record<string, string>,
): Promise<TokenResponse> {
  if (!AUTH_API_BASE_URL) {
    throw new Error("AUTH_API_BASE_URL chưa được cấu hình");
  }

  const response = await fetch(`${AUTH_API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error("Không thể cấp token");
  }
  return parseTokenResponse(await response.json());
}

async function loadUser(tokens: TokenResponse): Promise<User | null> {
  if (!AUTH_API_BASE_URL) {
    return null;
  }

  const response = await fetch(`${AUTH_API_BASE_URL}/me`, {
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${tokens.access_token}`,
    },
  });
  if (!response.ok) {
    return null;
  }

  const user = (await response.json()) as ApiUser;
  if (!user._id || !user.email || !user.role) {
    return null;
  }
  return {
    id: user._id,
    username: user.username,
    email: user.email,
    role: user.role,
    access_token: tokens.access_token,
    refresh_token: tokens.refresh_token,
    accessTokenExpires: getAccessTokenExpiry(tokens.access_token),
  };
}

async function refreshAccessToken(token: JWT): Promise<JWT> {
  try {
    if (!token.refresh_token) {
      throw new Error("Không có refresh token");
    }
    const tokens = await requestTokens("/refresh", {
      refresh_token: token.refresh_token,
    });
    return {
      ...token,
      access_token: tokens.access_token,
      refresh_token: tokens.refresh_token,
      accessTokenExpires: getAccessTokenExpiry(tokens.access_token),
      error: undefined,
    };
  } catch {
    return {
      ...token,
      error: "RefreshAccessTokenError",
    };
  }
}

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: {
          label: "Username",
          type: "text",
          placeholder: "Nguyen Van A",
        },
        password: { label: "Password", type: "password" },
        authorization_code: { label: "Authorization Code", type: "hidden" },
      },
      async authorize(credentials) {
        try {
          let tokens: TokenResponse;
          if (credentials?.authorization_code) {
            tokens = await requestTokens("/auth/oauth/exchange", {
              code: credentials.authorization_code,
            });
          } else if (credentials?.username && credentials.password) {
            tokens = await requestTokens("/login", {
              username: credentials.username,
              password: credentials.password,
            });
          } else {
            return null;
          }
          return await loadUser(tokens);
        } catch {
          return null;
        }
      },
    }),
  ],

  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.username = user.username;
        token.email = user.email;
        token.role = user.role;
        token.access_token = user.access_token;
        token.refresh_token = user.refresh_token;
        token.accessTokenExpires = user.accessTokenExpires;
        return token;
      }

      if (
        typeof token.accessTokenExpires === "number" &&
        Date.now() < token.accessTokenExpires
      ) {
        return token;
      }
      return refreshAccessToken(token);
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.username = token.username;
        session.user.email = token.email;
        session.user.id = token.id;
        session.user.role = token.role;
        session.user.access_token = token.access_token;
      }
      return session;
    },
  },

  session: {
    strategy: "jwt",
    maxAge: 60 * 60 * 24 * 7,
    updateAge: 60 * 60,
  },

  pages: {
    signIn: "/login",
  },

  secret: process.env.NEXTAUTH_SECRET,
};

export default NextAuth(authOptions);
