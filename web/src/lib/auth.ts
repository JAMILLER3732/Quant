import { createHash } from "node:crypto";
import { SignJWT, jwtVerify } from "jose";

const COOKIE_NAME = "admin_session";
const SESSION_TTL_SECONDS = 60 * 60 * 12; // 12 hours

// The session JWT is signed with a key derived from ADMIN_PASSWORD, so no
// second secret needs to be provisioned for a single-admin setup. If
// ADMIN_PASSWORD ever rotates, all existing sessions are invalidated for
// free as a side effect (the old signing key stops matching).
function signingKey(): Uint8Array {
  const password = process.env.ADMIN_PASSWORD;
  if (!password) {
    throw new Error("ADMIN_PASSWORD is not set — admin login is disabled until it is configured.");
  }
  return createHash("sha256").update(password).digest();
}

export function checkPassword(candidate: string): boolean {
  const password = process.env.ADMIN_PASSWORD;
  if (!password) return false;
  // Constant-time-ish comparison to avoid trivial timing side-channels.
  if (candidate.length !== password.length) return false;
  let diff = 0;
  for (let i = 0; i < password.length; i++) diff |= candidate.charCodeAt(i) ^ password.charCodeAt(i);
  return diff === 0;
}

export async function createSessionToken(): Promise<string> {
  return new SignJWT({ role: "admin" })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(`${SESSION_TTL_SECONDS}s`)
    .sign(signingKey());
}

export async function verifySessionToken(token: string | undefined): Promise<boolean> {
  if (!token) return false;
  try {
    const { payload } = await jwtVerify(token, signingKey());
    return payload.role === "admin";
  } catch {
    return false;
  }
}

export const SESSION_COOKIE = { name: COOKIE_NAME, maxAge: SESSION_TTL_SECONDS };
